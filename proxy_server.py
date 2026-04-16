import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError as RequestsConnectionError, ReadTimeout
from dotenv import load_dotenv
print("🔥 PROXY_SERVER KELOAD")
from fastapi import FastAPI, HTTPException

app = FastAPI()

print("🔥 APP STARTED")

@app.get("/")
def root():
    print("🔥 ROOT KEHIT")
    return {"status": "OK BRO"}
    
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from locate_text import (
    BBoxPx,
    QueryMode,
    decode_base64_image,
    encode_image_base64,
    detect_text,
    locate_text,
    warmup_easyocr,
    pad_bbox,
    pick_best_match,
    crop_image,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env")

DOCS_ROOT = ROOT.parent / "docs"

# EasyOCR model downloads may print unicode progress bars; make stdout/stderr UTF-8 on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DEBUG_TRACE = (os.getenv("VLM_DEBUG_TRACE") or "").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_STORE_LAST = (os.getenv("VLM_DEBUG_STORE_LAST") or "").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_MAX_RAW_CHARS = int((os.getenv("VLM_DEBUG_MAX_RAW_CHARS") or "4000").strip() or "4000")

_LAST_TRACE: dict[str, Any] | None = None


def _load_aksara_sunda_json_min() -> str:
    path = DOCS_ROOT / "aksara_sunda_full.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return ""


AKSARA_SUNDA_JSON_MIN = _load_aksara_sunda_json_min()

def _system_debug_meta() -> dict[str, Any]:
    # Jangan pernah log base64/api key. Ini hanya metadata prompt.
    aks_len = len(AKSARA_SUNDA_JSON_MIN or "")
    return {
        "aksara_json_loaded": bool(AKSARA_SUNDA_JSON_MIN),
        "aksara_json_len": aks_len,
        "system_len": len(SYSTEM),
        "system_prefix": SYSTEM[:220],
        "system_suffix": SYSTEM[-220:],
        "system_contains_aksara_json": bool(AKSARA_SUNDA_JSON_MIN) and (AKSARA_SUNDA_JSON_MIN[:32] in SYSTEM),
    }


def parse_first_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in response")
    return json.loads(m.group(0))


SYSTEM = (
    "Anda adalah interprater bahasa sunda yang bertindak sebagai OCR, anda WAJIB "
    "mengasumsikan bahwa gambar adalah aksara sunda lalu ke bahasa indonesia dan bukan sebaliknya. beberapa huruf, angka, "
    "dan simbol pasti menggambar kan tulisan bahasa sunda. Jangan menilai gambar "
    "sebagai pseudo script atau font.\n"
    "Langsung artikan tanpa menggunakan penjelasan apapun.\n"
    "Selalu gunakan ketikan aksara sunda dan bahasa indonesia setelahnya.\n"
    "Cara interpretasi gambar adalah menghitung jumlah symbol dan memperkirakan "
    "kata, huruf, dan angka tersebut.\n\n"
    "Aturan disambiguasi penting:\n"
    "- UTAMAKAN interpretasi simbol sebagai Aksara Sunda terlebih dahulu.\n"
    "- Jika simbol terlihat seperti angka/abjad latin (mis. '7', '77', '1', '0', dll), tetap anggap itu glyph Aksara Sunda,\n"
    "  kecuali sangat jelas konteksnya adalah angka (mis. tabel angka, format tanggal, atau deret digit murni).\n"
    "- Contoh: simbol mirip \"77\" jangan otomatis dibaca \"tujuh tujuh\"; utamakan sebagai huruf Sunda yang paling mungkin\n"
    "  (mis. bisa menjadi \"ᮊ\" = \"ka\").\n"
    "- Gunakan referensi JSON Unicode 1B80–1BBF di bawah untuk memilih glyph yang valid.\n\n"
    "Referensi aksara sunda (JSON, Unicode 1B80-1BBF):\n"
    + (AKSARA_SUNDA_JSON_MIN if AKSARA_SUNDA_JSON_MIN else "{}")
    + "\n\n"
    "Output WAJIB JSON dengan format:\n"
    '{ "aksara_sunda": "...", "aksara_sunda_indonesia": "..." }\n'
)


class OCRRequest(BaseModel):
    # base64 tanpa prefix data URI
    base64: str = Field(min_length=16)
    mime: str = Field(default="image/jpeg")
    prompt: str = Field(default="Baca teks pada gambar dan keluarkan JSON saja.")


class LocateTextRequest(BaseModel):
    base64: str = Field(min_length=16)
    mime: str = Field(default="image/jpeg")
    query: str = Field(min_length=1)
    queryMode: QueryMode = Field(default="contains")
    padPx: int = Field(default=6, ge=0, le=200)
    maxMatches: int = Field(default=10, ge=1, le=100)
    returnCrop: bool = Field(default=True)


class OcrFromCropRequest(BaseModel):
    base64: str = Field(min_length=16)
    mime: str = Field(default="image/jpeg")
    query: str = Field(min_length=1)
    queryMode: QueryMode = Field(default="contains")
    padPx: int = Field(default=6, ge=0, le=200)
    provider: str = Field(default="easyocr")
    prompt: str = Field(default="Baca teks pada gambar dan keluarkan JSON saja.")


class DetectTextRequest(BaseModel):
    base64: str = Field(min_length=16)
    mime: str = Field(default="image/jpeg")
    maxResults: int = Field(default=200, ge=1, le=1000)
    padPx: int = Field(default=6, ge=0, le=200)
    returnCrop: bool = Field(default=True)


class OcrAutoRequest(BaseModel):
    base64: str = Field(min_length=16)
    mime: str = Field(default="image/jpeg")
    prompt: str = Field(default="Baca teks pada gambar dan keluarkan JSON saja.")

_EASYOCR_WARMED = False


def _parse_bool_env(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


@app.on_event("startup")
async def _startup_warmup():
    """
    Pre-download / initialize EasyOCR models so playtests don't trigger downloads.
    """
    global _EASYOCR_WARMED
    if not _parse_bool_env("VLM_EASYOCR_PRELOAD", default=True):
        return
    try:
        await run_in_threadpool(warmup_easyocr)
        _EASYOCR_WARMED = True
    except Exception as e:
        # Don't block server startup; endpoints will still work once models are available.
        _EASYOCR_WARMED = False
        if DEBUG_TRACE:
            print(f"[warmup] EasyOCR warmup failed: {e}")

def _parse_cors_origins() -> list[str]:
    """
    Comma-separated origins for production deployments.
    Example:
      VLM_CORS_ORIGINS=https://your-site.netlify.app,https://your-custom-domain.com
    """
    raw = (os.getenv("VLM_CORS_ORIGINS") or "").strip()
    if not raw:
        return ["*"]
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


app.add_middleware(
    CORSMiddleware,
    # Untuk development/local file:// (origin: null) + WebView.
    # Untuk production: set VLM_CORS_ORIGINS (hindari "*" jika memungkinkan).
    allow_origins=_parse_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "easyocr_warmed": _EASYOCR_WARMED}


@app.post("/warmup")
def warmup():
    """
    Manual warmup endpoint (call once after server starts).
    """
    global _EASYOCR_WARMED
    try:
        warmup_easyocr()
        _EASYOCR_WARMED = True
        return {"ok": True, "easyocr_warmed": True}
    except Exception as e:
        _EASYOCR_WARMED = False
        raise HTTPException(status_code=500, detail=f"EasyOCR warmup failed: {e}")

@app.get("/debug/last")
def debug_last():
    if not DEBUG_STORE_LAST:
        raise HTTPException(
            status_code=400,
            detail="Enable VLM_DEBUG_STORE_LAST=1 in VLM/.env then restart server.",
        )
    return _LAST_TRACE or {"empty": True}


def _store_trace(trace: dict[str, Any]) -> None:
    global _LAST_TRACE
    if not DEBUG_STORE_LAST:
        return
    _LAST_TRACE = trace


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(truncated {len(text) - limit} chars)"


def _should_retry_openrouter_error(e: Exception) -> bool:
    msg = str(e) or ""
    if "Response ended prematurely" in msg:
        return True
    if isinstance(e, (ChunkedEncodingError, RequestsConnectionError, ReadTimeout)):
        return True
    return False


def _openrouter_post_with_retry(*, url: str, headers: dict[str, str], body: dict[str, Any]) -> requests.Response:
    max_attempts = int((os.getenv("VLM_OPENROUTER_RETRY_MAX") or "3").strip() or "3")
    base_sleep = float((os.getenv("VLM_OPENROUTER_RETRY_SLEEP_S") or "0.6").strip() or "0.6")
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return requests.post(url, headers=headers, json=body, timeout=120)
        except Exception as e:
            last_exc = e
            if not _should_retry_openrouter_error(e) or attempt >= max_attempts:
                raise
            time.sleep(base_sleep * attempt)
    raise last_exc or RuntimeError("OpenRouter retry failed")


@app.post("/ocr/openrouter")
def ocr_openrouter(req: OCRRequest):
    api_key = os.getenv("OPENROUTER_API_KEY") or ""
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY missing")
    model = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it")

    url = "https://openrouter.ai/api/v1/chat/completions"
    data_uri = f"data:{req.mime};base64,{req.base64}"

    body = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": req.prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
    }

    t0 = time.perf_counter()
    try:
        r = _openrouter_post_with_retry(
            url=url,
            headers={"Authorization": f"Bearer {api_key}"},
            body=body,
        )
    except Exception as e:
        # Retryable errors: Response ended prematurely, connection reset, timeouts, etc.
        _store_trace(
            {
                "provider": "openrouter",
                "model": model,
                "ms": int((time.perf_counter() - t0) * 1000),
                "status": "exception",
                "request": {
                    "mime": req.mime,
                    "base64_len": len(req.base64),
                    "prompt": req.prompt,
                    "system": _system_debug_meta(),
                },
                "error": _shorten(str(e), DEBUG_MAX_RAW_CHARS),
            }
        )
        raise HTTPException(status_code=502, detail=f"OpenRouter request failed: {e}")
    dt_ms = int((time.perf_counter() - t0) * 1000)
    if r.status_code >= 400:
        _store_trace(
            {
                "provider": "openrouter",
                "model": model,
                "ms": dt_ms,
                "status": r.status_code,
                "request": {
                    "mime": req.mime,
                    "base64_len": len(req.base64),
                    "prompt": req.prompt,
                    "system": _system_debug_meta(),
                },
                "error": _shorten(r.text, DEBUG_MAX_RAW_CHARS),
            }
        )
        raise HTTPException(status_code=r.status_code, detail=r.text)
    data = r.json()
    text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
    try:
        parsed = parse_first_json_object(text)
    except Exception as e:
        _store_trace(
            {
                "provider": "openrouter",
                "model": model,
                "ms": dt_ms,
                "status": 200,
                "request": {
                    "mime": req.mime,
                    "base64_len": len(req.base64),
                    "prompt": req.prompt,
                    "system": _system_debug_meta(),
                },
                "response_raw": _shorten(text, DEBUG_MAX_RAW_CHARS),
                "parse_error": str(e),
            }
        )
        raise HTTPException(status_code=500, detail=f"JSON parse failed: {e}; raw={text!r}")

    trace = {
        "provider": "openrouter",
        "model": model,
        "ms": dt_ms,
        "status": 200,
        "request": {
            "mime": req.mime,
            "base64_len": len(req.base64),
            "prompt": req.prompt,
            "system": _system_debug_meta(),
        },
        "response_raw": _shorten(text, DEBUG_MAX_RAW_CHARS),
        "parsed": parsed,
    }
    _store_trace(trace)
    if DEBUG_TRACE:
        # Print ringkas ke stdout (tanpa base64, tanpa API key)
        print(json.dumps(trace, ensure_ascii=False))
    return {"raw": text, "json": parsed}


@app.post("/ocr/ollama")
def ocr_ollama(req: OCRRequest):
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    model = os.getenv("OLLAMA_MODEL", "gemma4")

    # Ollama expects raw base64 (no data URI)
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": req.prompt, "images": [req.base64]},
        ],
        "options": {"temperature": 0.2},
    }

    t0 = time.perf_counter()
    r = requests.post(url, json=body, timeout=120)
    dt_ms = int((time.perf_counter() - t0) * 1000)
    if r.status_code >= 400:
        _store_trace(
            {
                "provider": "ollama",
                "model": model,
                "ms": dt_ms,
                "status": r.status_code,
                "request": {
                    "mime": req.mime,
                    "base64_len": len(req.base64),
                    "prompt": req.prompt,
                    "system": _system_debug_meta(),
                    "url": url,
                },
                "error": _shorten(r.text, DEBUG_MAX_RAW_CHARS),
            }
        )
        raise HTTPException(status_code=r.status_code, detail=r.text)
    data = r.json()
    text = (data.get("message") or {}).get("content") or ""
    try:
        parsed = parse_first_json_object(text)
    except Exception as e:
        _store_trace(
            {
                "provider": "ollama",
                "model": model,
                "ms": dt_ms,
                "status": 200,
                "request": {
                    "mime": req.mime,
                    "base64_len": len(req.base64),
                    "prompt": req.prompt,
                    "system": _system_debug_meta(),
                    "url": url,
                },
                "response_raw": _shorten(text, DEBUG_MAX_RAW_CHARS),
                "parse_error": str(e),
            }
        )
        raise HTTPException(status_code=500, detail=f"JSON parse failed: {e}; raw={text!r}")
    trace = {
        "provider": "ollama",
        "model": model,
        "ms": dt_ms,
        "status": 200,
        "request": {
            "mime": req.mime,
            "base64_len": len(req.base64),
            "prompt": req.prompt,
            "system": _system_debug_meta(),
            "url": url,
        },
        "response_raw": _shorten(text, DEBUG_MAX_RAW_CHARS),
        "parsed": parsed,
    }
    _store_trace(trace)
    if DEBUG_TRACE:
        print(json.dumps(trace, ensure_ascii=False))
    return {"raw": text, "json": parsed}


@app.post("/locate-text")
def locate_text_api(req: LocateTextRequest):
    try:
        img = decode_base64_image(req.base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image base64: {e}")

    try:
        matches = locate_text(
            img=img,
            query=req.query,
            mode=req.queryMode,
            max_matches=req.maxMatches,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR locate failed: {e}")

    best = pick_best_match(matches)
    crop_b64: str | None = None
    best_out = None
    if best is not None:
        padded = pad_bbox(best.bbox, req.padPx, img.width, img.height)
        if req.returnCrop and padded.w > 0 and padded.h > 0:
            try:
                crop = crop_image(img, padded)
                crop_b64 = encode_image_base64(crop, mime=req.mime)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Crop encode failed: {e}")
        best_out = {
            "text": best.text,
            "confidence": best.confidence,
            "source": best.source,
            "bboxPx": {"x": padded.x, "y": padded.y, "w": padded.w, "h": padded.h},
            "bboxNorm": {
                "x": (padded.x / img.width) if img.width else 0,
                "y": (padded.y / img.height) if img.height else 0,
                "w": (padded.w / img.width) if img.width else 0,
                "h": (padded.h / img.height) if img.height else 0,
            },
        }

    def _match_out(m):
        b: BBoxPx = m.bbox
        return {
            "text": m.text,
            "confidence": m.confidence,
            "source": m.source,
            "bboxPx": {"x": b.x, "y": b.y, "w": b.w, "h": b.h},
            "bboxNorm": {
                "x": (b.x / img.width) if img.width else 0,
                "y": (b.y / img.height) if img.height else 0,
                "w": (b.w / img.width) if img.width else 0,
                "h": (b.h / img.height) if img.height else 0,
            },
        }

    return {
        "width": img.width,
        "height": img.height,
        "query": req.query,
        "queryMode": req.queryMode,
        "padPx": req.padPx,
        "matches": [_match_out(m) for m in matches],
        "bestMatch": best_out,
        "cropBase64": crop_b64,
        "cropMime": req.mime if crop_b64 else None,
    }


@app.post("/detect-text")
def detect_text_api(req: DetectTextRequest):
    try:
        img = decode_base64_image(req.base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image base64: {e}")

    try:
        matches = detect_text(img=img, max_results=req.maxResults)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text detection failed: {e}")

    crop_b64: str | None = None
    crop_bbox: BBoxPx | None = None
    if req.returnCrop and matches:
        try:
            x1 = min(m.bbox.x for m in matches)
            y1 = min(m.bbox.y for m in matches)
            x2 = max(m.bbox.x + m.bbox.w for m in matches)
            y2 = max(m.bbox.y + m.bbox.h for m in matches)
            union = BBoxPx(x=x1, y=y1, w=max(0, x2 - x1), h=max(0, y2 - y1))
            padded = pad_bbox(union, req.padPx, img.width, img.height)
            crop_bbox = padded
            if padded.w > 0 and padded.h > 0:
                crop = crop_image(img, padded)
                crop_b64 = encode_image_base64(crop, mime=req.mime)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text crop failed: {e}")

    def _match_out(m):
        b: BBoxPx = m.bbox
        return {
            "text": m.text,
            "confidence": m.confidence,
            "source": m.source,
            "bboxPx": {"x": b.x, "y": b.y, "w": b.w, "h": b.h},
            "bboxNorm": {
                "x": (b.x / img.width) if img.width else 0,
                "y": (b.y / img.height) if img.height else 0,
                "w": (b.w / img.width) if img.width else 0,
                "h": (b.h / img.height) if img.height else 0,
            },
        }

    return {
        "width": img.width,
        "height": img.height,
        "matches": [_match_out(m) for m in matches],
        "cropBase64": crop_b64,
        "cropMime": req.mime if crop_b64 else None,
        "cropBboxPx": (
            {"x": crop_bbox.x, "y": crop_bbox.y, "w": crop_bbox.w, "h": crop_bbox.h} if crop_bbox else None
        ),
    }


@app.post("/ocr/auto")
def ocr_auto(req: OcrAutoRequest):
    provider = (os.getenv("VLM_OCR_PROVIDER") or "").strip().lower()
    if not provider:
        provider = "openrouter" if (os.getenv("OPENROUTER_API_KEY") or "").strip() else "ollama"
    ocr_req = OCRRequest(base64=req.base64, mime=req.mime, prompt=req.prompt)
    try:
        if provider == "openrouter":
            out = ocr_openrouter(ocr_req)
        else:
            out = ocr_ollama(ocr_req)
        # Tell client which provider was used (for UI indicators).
        return {"provider": provider, **out}
    except HTTPException:
        raise
    except Exception as e:
        # Example: Ollama not running / connection refused
        raise HTTPException(status_code=502, detail=f"OCR provider '{provider}' failed: {e}")


@app.post("/ocr/from-crop")
def ocr_from_crop(req: OcrFromCropRequest):
    locate_req = LocateTextRequest(
        base64=req.base64,
        mime=req.mime,
        query=req.query,
        queryMode=req.queryMode,
        padPx=req.padPx,
        maxMatches=20,
        returnCrop=True,
    )
    located = locate_text_api(locate_req)
    crop_b64 = located.get("cropBase64")
    if not crop_b64:
        return {
            "located": located,
            "ocr": None,
            "error": "NO_CROP_MATCH",
        }

    ocr_req = OCRRequest(base64=crop_b64, mime=req.mime, prompt=req.prompt)
    provider = (req.provider or "").strip().lower()
    if provider == "openrouter":
        ocr = ocr_openrouter(ocr_req)
    else:
        ocr = ocr_ollama(ocr_req)
    return {"located": located, "ocr": ocr}
    
import uvicorn

if __name__ == "__main__":
    uvicorn.run("proxy_server:app", host="0.0.0.0", port=8000)
