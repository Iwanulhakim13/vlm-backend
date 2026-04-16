# ===== IMPORT =====
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

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

# ===== INIT =====
app = FastAPI()
ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env")

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODELS =====
class OCRRequest(BaseModel):
    base64: str = Field(min_length=16)
    mime: str = Field(default="image/jpeg")
    prompt: str = Field(default="Baca teks pada gambar.")

class OcrAutoRequest(BaseModel):
    base64: str = Field(min_length=16)
    mime: str = Field(default="image/jpeg")
    prompt: str = Field(default="Baca teks pada gambar.")

class LocateTextRequest(BaseModel):
    base64: str
    mime: str = "image/jpeg"
    query: str
    queryMode: QueryMode = "contains"
    padPx: int = 6
    maxMatches: int = 10
    returnCrop: bool = True

class OcrFromCropRequest(BaseModel):
    base64: str
    mime: str = "image/jpeg"
    query: str
    queryMode: QueryMode = "contains"
    padPx: int = 6
    provider: str = "easyocr"
    prompt: str = "Baca teks pada gambar."

@app.on_event("startup")
async def startup():
    print("Server started (no warmup)")

@app.get("/")
def root():
    return {"status": "OK"}

# ===== EASY OCR CORE =====
def run_easyocr(base64_str, mime):
    img = decode_base64_image(base64_str)
    matches = detect_text(img=img)

    text = " ".join([m.text for m in matches]) if matches else ""

    return {
        "raw": text,
        "json": {
            "aksara_sunda": text,
            "aksara_sunda_indonesia": text
        }
    }

# ===== OPENROUTER (OPSIONAL) =====
def ocr_openrouter(req: OCRRequest):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(500, "OPENROUTER_API_KEY missing")

    url = "https://openrouter.ai/api/v1/chat/completions"

    data_uri = f"data:{req.mime};base64,{req.base64}"

    body = {
        "model": "google/gemma-4-26b-a4b-it",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": req.prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]}
        ]
    }

    r = requests.post(url, headers={
        "Authorization": f"Bearer {api_key}"
    }, json=body)

    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)

    return r.json()

# ===== OCR AUTO (FIX UTAMA) =====
@app.post("/ocr/auto")
def ocr_auto(req: OcrAutoRequest):

    provider = (os.getenv("VLM_OCR_PROVIDER") or "").lower()

    # 🔥 FIX: default easyocr
    if not provider:
        provider = "easyocr"

    ocr_req = OCRRequest(**req.dict())

    try:
        if provider == "openrouter":
            result = ocr_openrouter(ocr_req)

        elif provider == "easyocr":
            result = run_easyocr(req.base64, req.mime)

        else:
            raise HTTPException(400, f"Provider '{provider}' tidak didukung")

        return {
            "provider": provider,
            **result
        }

    except Exception as e:
        raise HTTPException(502, f"OCR ERROR: {e}")

# ===== LOCATE TEXT =====
@app.post("/locate-text")
def locate_text_api(req: LocateTextRequest):
    img = decode_base64_image(req.base64)
    matches = locate_text(img=img, query=req.query)

    return {"matches": [m.text for m in matches]}

# ===== OCR FROM CROP =====
@app.post("/ocr/from-crop")
def ocr_from_crop(req: OcrFromCropRequest):

    img = decode_base64_image(req.base64)
    matches = locate_text(img=img, query=req.query)

    if not matches:
        return {"error": "NO_MATCH"}

    best = matches[0]
    crop = crop_image(img, best.bbox)
    crop_b64 = encode_image_base64(crop)

    if req.provider == "openrouter":
        result = ocr_openrouter(OCRRequest(base64=crop_b64))

    else:
        result = run_easyocr(crop_b64, req.mime)

    return result
