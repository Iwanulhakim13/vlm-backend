import base64
import io
import os
import re
from dataclasses import dataclass
from typing import Iterable, Literal

from PIL import Image


QueryMode = Literal["contains", "exact", "regex"]


@dataclass(frozen=True)
class BBoxPx:
    x: int
    y: int
    w: int
    h: int

    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)


@dataclass(frozen=True)
class OCRToken:
    text: str
    conf: float
    bbox: BBoxPx
    line_key: tuple[int, int, int, int]  # (page, block, par, line)


@dataclass(frozen=True)
class LocateMatch:
    text: str
    confidence: float
    bbox: BBoxPx
    source: str  # "token" | "phrase" | "line"


def decode_base64_image(base64_str: str) -> Image.Image:
    raw = base64.b64decode(base64_str, validate=False)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def encode_image_base64(img: Image.Image, mime: str = "image/jpeg", quality: int = 85) -> str:
    fmt = "JPEG" if mime.lower() in {"image/jpg", "image/jpeg"} else "PNG"
    out = io.BytesIO()
    if fmt == "JPEG":
        img.save(out, format=fmt, quality=quality, optimize=True)
    else:
        img.save(out, format=fmt, optimize=True)
    return base64.b64encode(out.getvalue()).decode("ascii")


def clamp_bbox(b: BBoxPx, img_w: int, img_h: int) -> BBoxPx:
    x1 = max(0, min(img_w, b.x))
    y1 = max(0, min(img_h, b.y))
    x2 = max(0, min(img_w, b.x + b.w))
    y2 = max(0, min(img_h, b.y + b.h))
    return BBoxPx(x=x1, y=y1, w=max(0, x2 - x1), h=max(0, y2 - y1))


def pad_bbox(b: BBoxPx, pad_px: int, img_w: int, img_h: int) -> BBoxPx:
    if pad_px <= 0:
        return clamp_bbox(b, img_w, img_h)
    return clamp_bbox(
        BBoxPx(x=b.x - pad_px, y=b.y - pad_px, w=b.w + 2 * pad_px, h=b.h + 2 * pad_px),
        img_w,
        img_h,
    )


def crop_image(img: Image.Image, bbox: BBoxPx) -> Image.Image:
    b = clamp_bbox(bbox, img.width, img.height)
    return img.crop((b.x, b.y, b.x + b.w, b.y + b.h))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _tokens_to_line_text(tokens: Iterable[OCRToken]) -> str:
    parts = []
    for t in tokens:
        if t.text:
            parts.append(t.text)
    return " ".join(parts)


_EASY_READER = None


def _parse_bool_env(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


def _get_easyocr_reader():
    global _EASY_READER
    if _EASY_READER is not None:
        return _EASY_READER
    try:
        import easyocr  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"easyocr not installed: {e}")
    # lang list kept minimal; add more if needed
    use_gpu = _parse_bool_env("VLM_EASYOCR_GPU", default=False)
    _EASY_READER = easyocr.Reader(["en", "id"], gpu=use_gpu)
    return _EASY_READER


def warmup_easyocr() -> None:
    """
    Force EasyOCR to initialize (and download models if missing) ahead of playtests.
    """
    reader = _get_easyocr_reader()
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"numpy not installed (required by easyocr): {e}")
    img = Image.new("RGB", (32, 32), "white")
    _ = reader.readtext(np.array(img))


def ocr_tokens_craft(img: Image.Image) -> list[OCRToken]:
    """
    EasyOCR uses CRAFT for text detection, returning word boxes + text + confidence.
    """
    reader = _get_easyocr_reader()
    # readtext accepts numpy array; convert via PIL->RGB bytes to avoid cv2 dependency here
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"numpy not installed (required by easyocr): {e}")

    arr = np.array(img.convert("RGB"))
    results = reader.readtext(arr)  # [(bbox, text, conf)]
    out: list[OCRToken] = []
    for item in results:
        if not item or len(item) < 3:
            continue
        bbox_pts, text, conf = item[0], item[1], item[2]
        text = (text or "").strip()
        if not text:
            continue
        try:
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        except Exception:
            continue
        bbox = BBoxPx(x=x1, y=y1, w=max(0, x2 - x1), h=max(0, y2 - y1))
        # EasyOCR doesn't provide stable line ids; keep a dummy line_key
        out.append(OCRToken(text=text, conf=float(conf) * 100.0, bbox=bbox, line_key=(0, 0, 0, 0)))
    return out


def detect_text(
    *,
    img: Image.Image,
    max_results: int = 200,
) -> list[LocateMatch]:
    tokens = ocr_tokens_craft(img)
    matches: list[LocateMatch] = []
    for t in tokens[: max(0, max_results)]:
        matches.append(LocateMatch(text=t.text, confidence=t.conf, bbox=t.bbox, source="token"))
    return matches


def locate_text(
    *,
    img: Image.Image,
    query: str,
    mode: QueryMode = "contains",
    max_matches: int = 10,
) -> list[LocateMatch]:
    q = (query or "").strip()
    if not q:
        return []

    tokens = ocr_tokens_craft(img)
    qn = _norm(q)
    matches: list[LocateMatch] = []

    if mode in {"contains", "exact"}:
        for t in tokens:
            tn = _norm(t.text)
            ok = (qn == tn) if mode == "exact" else (qn in tn)
            if ok:
                matches.append(
                    LocateMatch(text=t.text, confidence=t.conf, bbox=t.bbox, source="token")
                )
                if len(matches) >= max_matches:
                    return matches

        # Phrase match within the same line: slide over tokens.
        q_words = [w for w in qn.split(" ") if w]
        if len(q_words) >= 2:
            by_line: dict[tuple[int, int, int, int], list[OCRToken]] = {}
            for t in tokens:
                by_line.setdefault(t.line_key, []).append(t)

            for line_tokens in by_line.values():
                line_norms = [_norm(t.text) for t in line_tokens]
                for i in range(0, max(0, len(line_norms) - len(q_words) + 1)):
                    window = line_norms[i : i + len(q_words)]
                    ok = (window == q_words) if mode == "exact" else (" ".join(q_words) in " ".join(window))
                    if not ok:
                        continue
                    bb = _union_bbox([t.bbox for t in line_tokens[i : i + len(q_words)]])
                    conf = _avg_conf([t.conf for t in line_tokens[i : i + len(q_words)]])
                    text = _tokens_to_line_text(line_tokens[i : i + len(q_words)])
                    matches.append(LocateMatch(text=text, confidence=conf, bbox=bb, source="phrase"))
                    if len(matches) >= max_matches:
                        return matches

    if mode == "regex":
        rx = re.compile(q, flags=re.IGNORECASE)
        # Token-level regex
        for t in tokens:
            if rx.search(t.text or ""):
                matches.append(LocateMatch(text=t.text, confidence=t.conf, bbox=t.bbox, source="token"))
                if len(matches) >= max_matches:
                    return matches

        # Line-level fallback (bbox = entire line)
        by_line: dict[tuple[int, int, int, int], list[OCRToken]] = {}
        for t in tokens:
            by_line.setdefault(t.line_key, []).append(t)
        for line_tokens in by_line.values():
            line_text = _tokens_to_line_text(line_tokens)
            if not line_text:
                continue
            if rx.search(line_text):
                bb = _union_bbox([t.bbox for t in line_tokens])
                conf = _avg_conf([t.conf for t in line_tokens])
                matches.append(LocateMatch(text=line_text, confidence=conf, bbox=bb, source="line"))
                if len(matches) >= max_matches:
                    return matches

    return matches[:max_matches]


def _avg_conf(values: Iterable[float]) -> float:
    xs = [v for v in values if v is not None and v >= 0]
    if not xs:
        return -1.0
    return float(sum(xs) / len(xs))


def _union_bbox(boxes: Iterable[BBoxPx]) -> BBoxPx:
    xs = list(boxes)
    if not xs:
        return BBoxPx(0, 0, 0, 0)
    x1 = min(b.x for b in xs)
    y1 = min(b.y for b in xs)
    x2 = max(b.x + b.w for b in xs)
    y2 = max(b.y + b.h for b in xs)
    return BBoxPx(x=x1, y=y1, w=max(0, x2 - x1), h=max(0, y2 - y1))


def pick_best_match(matches: list[LocateMatch]) -> LocateMatch | None:
    if not matches:
        return None
    # Prefer higher confidence, then larger area.
    return sorted(matches, key=lambda m: (m.confidence, m.bbox.area()), reverse=True)[0]

