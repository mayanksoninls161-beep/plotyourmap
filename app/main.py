"""
Adaptive Booth Deployment API.

Same endpoints as the production "main" API, with the hall+booth endpoint
upgraded to run OUR adaptive booth pipeline instead of the old single-pass
EnsembleDetector:

  POST /predict                  icons + trails (Roboflow)            [unchanged]
  POST /debug_predict            trail debug (Roboflow)               [unchanged]
  POST /hall_with_booth_predict  Roboflow hall detection  +  ADAPTIVE booth
                                 pipeline (auto profile -> tiling + big-region +
                                 PDF vector / OCR labels). Accepts image_url
                                 AND/OR pdf_url; when BOTH are given everything is
                                 computed from the PDF.                [UPGRADED]

Layout (works locally and in the Docker image):
  <here>/pipeline   -> baked production detector package (detectors/, utils/)
                       plus support modules (logging_setup, trail_merger,
                       image_hash_checker).
  <here>/adaptive   -> the adaptive engine (pipeline.py, config.py, tiling.py,
                       labeling.py, input_profile.py, _detectors.py, _calib.py).

S3 writeback is hard-OFF (constraint). Roboflow models load lazily on first use
so the app boots even without keys; each endpoint only needs its own Roboflow
key when it is actually called.
"""
from __future__ import annotations

import os
import sys

# ── Make the baked engine + detector package importable BEFORE other imports ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.join(_HERE, "pipeline")    # detectors, utils, support modules
_ADAPT = os.path.join(_HERE, "adaptive")   # adaptive engine (flat modules)
# Insert _PIPE first, then _ADAPT, so the FINAL order is [_ADAPT, _PIPE, ...]:
# the adaptive flat modules (config/labeling/tiling/input_profile) must take
# priority over any identically-named installed package.
for _p in (_PIPE, _ADAPT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Tell the adaptive _detectors bridge where the baked detector package lives.
os.environ.setdefault("BOOTH_DETECTOR_ROOT", _PIPE)

import asyncio
import glob
import hashlib
import json
import logging
import shutil
import tempfile
from datetime import datetime
from io import BytesIO

import cv2
import httpx
import imagehash
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Security, UploadFile, status
from fastapi.security import APIKeyHeader
from PIL import Image

# Support modules (from _PIPE)
from logging_setup import setup_logging
from trail_merger import merge_trails
from image_hash_checker import (
    content_type_for_bytes,
    load_hash_db,
    parse_s3_url,
    read_image_metadata,
    save_hash_db,
    upload_bytes_to_s3,
    write_image_metadata,
)

# Adaptive booth engine (from _ADAPT). `pipeline` here is adaptive/pipeline.py.
import pipeline as adaptive_pipeline

load_dotenv()
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

_LOG_PATH = setup_logging()
logger = logging.getLogger(__name__)
logger.info("Adaptive booth API starting; deep logs at %s", _LOG_PATH)

# ─────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────
_API_KEY = os.getenv("AUTHENTICATION_API_KEY")
if not _API_KEY:
    raise ValueError("AUTHENTICATION_API_KEY environment variable is not set")

_api_key_header = APIKeyHeader(name="Authentication-API-Key", auto_error=False)


async def require_api_key(key: str = Security(_api_key_header)):
    if key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key or Add API key in the Header",
        )


app = FastAPI(title="Adaptive Booth API")

# ─────────────────────────────────────────────────────────────────
# MODEL SETUP  (lazy: download/connect on first use, not at import)
# ─────────────────────────────────────────────────────────────────
_MODEL_SPECS = {
    "icon": ("ROBOFLOW_ICON_API_KEY", "plotmymap_synthetic/2"),
    "trail": ("ROBOFLOW_TRAIL_API_KEY", "plotourmap-trails-largedataset/3"),
    "hall": ("ROBOFLOW_HALL_API_KEY", "hall_detection/6"),
}
_models: dict = {}


def _get_rf_model(kind: str):
    """Lazily construct (and cache) a Roboflow model. Raises 500 if its key is
    missing so booth-only endpoints can still run without Roboflow keys."""
    if kind not in _models:
        env_name, model_id = _MODEL_SPECS[kind]
        key = os.getenv(env_name)
        if not key:
            raise HTTPException(status_code=500,
                                detail=f"{env_name} is not set")
        from inference import get_model
        logger.info("Loading Roboflow model %s (%s)", kind, model_id)
        _models[kind] = get_model(model_id=model_id, api_key=key)
    return _models[kind]


http_client = httpx.AsyncClient(timeout=60)

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
TRAIL_MERGE_CONFIG = {"cross_cluster_merge": True}
HASH_CHECK_MODE = "always_run"  # "always_run" | "skip_if_present" | "reject_if_present"
S3_WRITEBACK_ENABLED = os.getenv("S3_WRITEBACK_ENABLED", "false").lower() == "true"
HALL_RASTER_MAX_EDGE = int(os.getenv("HALL_RASTER_MAX_EDGE", "2048"))

PERSIST_ENABLED = os.getenv("PERSIST_EXECUTIONS", "true").lower() == "true"
PERSIST_IN_DIR = os.getenv("PERSIST_IN_DIR", "/data/in")
PERSIST_OUT_DIR = os.getenv("PERSIST_OUT_DIR", "/data/out")

_hash_db_lock = asyncio.Lock()


def _persist_execution(endpoint, input_bytes, input_name, result) -> None:
    if not PERSIST_ENABLED:
        return
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe = "".join(c for c in (input_name or "input")
                       if c.isalnum() or c in ("-", "_", ".")) or "input"
        base = f"{ts}_{endpoint}_{safe}"
        os.makedirs(PERSIST_IN_DIR, exist_ok=True)
        os.makedirs(PERSIST_OUT_DIR, exist_ok=True)
        if input_bytes is not None:
            with open(os.path.join(PERSIST_IN_DIR, base), "wb") as fh:
                fh.write(input_bytes)
        with open(os.path.join(PERSIST_OUT_DIR, base + ".json"), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, default=str, indent=2)
    except Exception:
        logger.exception("Failed to persist execution for %s", endpoint)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def _aws_creds_present() -> bool:
    return bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))


def _s3_download_bytes(bucket: str, key: str, region: str) -> bytes:
    """Read-only download of a private S3 object using the AWS creds in the
    environment (loaded from .env). Does NOT touch the writeback path."""
    import boto3
    region = region or os.getenv("AWS_REGION") or None
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


async def fetch_image(url: str) -> BytesIO:
    """Fetch bytes for a URL. If the URL resolves to an S3 object and AWS creds
    are configured, download it via boto3 (so PRIVATE s3:// or https S3 URLs work
    without presigning); otherwise fall back to a plain HTTPS GET (public or
    presigned URLs, incl. CloudFront)."""
    loop = asyncio.get_event_loop()

    # Normalize: tolerate scheme-less URLs (e.g. "host/key" or "//host/key")
    # that frontends/forms sometimes send -> httpx needs an explicit scheme.
    url = (url or "").strip()
    if url and "://" not in url:
        url = "https://" + url.lstrip("/")

    # 1) Credentialed S3 download for private S3 / s3:// URLs.
    parsed = parse_s3_url(url) if _aws_creds_present() else None
    if parsed:
        bucket, key, region = parsed
        try:
            data = await loop.run_in_executor(
                None, _s3_download_bytes, bucket, key, region)
            return BytesIO(data)
        except Exception as e:
            logger.warning("S3 credentialed fetch failed for %s/%s (%s); "
                           "falling back to plain HTTPS", bucket, key, e)

    # 2) Plain HTTPS GET (public or presigned URLs).
    try:
        async with http_client.stream("GET", url) as response:
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch image")
            buf = BytesIO()
            async for chunk in response.aiter_bytes():
                buf.write(chunk)
            buf.seek(0)
            return buf
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image fetch failed: {e}")


async def run_model_async(model, image_data, confidence: float):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: model.infer(image_data, confidence=confidence))


def compute_hash_from_bytes(raw_bytes: bytes) -> str:
    img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    return str(imagehash.phash(img))


async def get_or_compute_hash(raw_bytes: bytes, extra: dict = None):
    loop = asyncio.get_event_loop()
    existing_hash, existing_date = await loop.run_in_executor(
        None, read_image_metadata, raw_bytes)
    if existing_hash and existing_date:
        return existing_hash, existing_date, raw_bytes, False
    phash = await loop.run_in_executor(None, compute_hash_from_bytes, raw_bytes)
    updated_bytes = await loop.run_in_executor(
        None, lambda: write_image_metadata(raw_bytes, phash, None, extra))
    _, written_date = await loop.run_in_executor(
        None, read_image_metadata, updated_bytes)
    was_written = updated_bytes is not raw_bytes and len(updated_bytes) != len(raw_bytes)
    return phash, written_date, updated_bytes, was_written


async def check_and_register_hash(img_hash: str) -> bool:
    loop = asyncio.get_event_loop()
    async with _hash_db_lock:
        hash_db = await loop.run_in_executor(None, load_hash_db)
        if img_hash in hash_db:
            return True
        hash_db.add(img_hash)
        await loop.run_in_executor(None, save_hash_db, hash_db)
        return False


async def writeback_to_s3(image_url: str, modified_bytes: bytes) -> dict:
    if not S3_WRITEBACK_ENABLED:
        return {"attempted": False, "reason": "writeback disabled"}
    parsed = parse_s3_url(image_url)
    if not parsed:
        return {"attempted": False, "reason": "URL not resolvable to S3"}
    bucket, key, region = parsed
    ctype = content_type_for_bytes(modified_bytes)
    loop = asyncio.get_event_loop()
    upload_result = await loop.run_in_executor(
        None, upload_bytes_to_s3, bucket, key, modified_bytes, region, ctype)
    response = {"attempted": True, "success": upload_result["success"],
                "bucket": bucket, "key": key, "region": region,
                "content_type": ctype}
    if not upload_result["success"]:
        response["error"] = upload_result["error"]
        response["error_code"] = upload_result["error_code"]
    return response


def _deep_serialize(obj):
    if isinstance(obj, dict):
        return {k: _deep_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_serialize(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return {k: _deep_serialize(v) for k, v in obj.__dict__.items()
                if not k.startswith("_")}
    return obj


def serialize_model_output(model_output) -> list:
    if model_output is None:
        return []
    if isinstance(model_output, list) and all(isinstance(x, dict) for x in model_output):
        return model_output
    items = model_output if isinstance(model_output, list) else [model_output]
    serialized = []
    for item in items:
        for method in ["model_dump", "dict"]:
            if hasattr(item, method):
                try:
                    serialized.append(getattr(item, method)())
                    break
                except Exception:
                    continue
        else:
            if hasattr(item, "json"):
                try:
                    serialized.append(json.loads(item.json()))
                    continue
                except Exception:
                    pass
            if hasattr(item, "__dict__"):
                try:
                    serialized.append(_deep_serialize(item.__dict__))
                    continue
                except Exception:
                    pass
            if isinstance(item, dict):
                serialized.append(item)
            else:
                logger.warning(f"Could not serialize: {type(item)}")
    return serialized


def extract_trail_data(serialized_output: list):
    if not serialized_output:
        return {}, []
    block = serialized_output[0] if isinstance(serialized_output, list) else serialized_output
    if isinstance(block, dict):
        preds = block.get("predictions", [])
    else:
        preds = []
        block = {}
    return block, preds


def log_predictions(predictions: list, label: str):
    total = len(predictions)
    with_points = sum(1 for p in predictions if p.get("points"))
    logger.info(f"[{label}] {total} predictions, {with_points} with points")


# ─────────────────────────────────────────────────────────────────
# ADAPTIVE BOOTH ENGINE INTEGRATION
# ─────────────────────────────────────────────────────────────────
def _safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name or "input"))[0] or "input"
    return "".join(c for c in stem if c.isalnum() or c in ("-", "_", ".")) or "input"


def _looks_like_pdf(b: bytes) -> bool:
    return bool(b) and b[:4] == b"%PDF"


def _run_adaptive_booths(src_bytes: bytes, src_name: str, is_pdf: bool,
                         page: int, fp_policy, want_render=True, want_viz=False):
    """Run the adaptive pipeline on raw bytes (PDF or image).

    Returns (payload, render_bgr, viz_png_bytes). `render_bgr` is the page render
    in the SAME pixel space as the returned booth coordinates (used to align the
    Roboflow hall boxes). Everything runs inside a temp dir that is removed here,
    so the render is read into memory before cleanup.
    """
    workdir = tempfile.mkdtemp(prefix="adaptive_")
    try:
        stem = _safe_stem(src_name)
        ext = ".pdf" if is_pdf else (os.path.splitext(src_name or "")[1] or ".png")
        in_path = os.path.join(workdir, stem + ext)
        with open(in_path, "wb") as fh:
            fh.write(src_bytes)
        outdir = os.path.join(workdir, "out")

        payload = adaptive_pipeline.run(in_path, outdir, page_index=page,
                                        fp_policy=fp_policy, verbose=False)

        render_bgr = None
        viz_png = None
        rdir = os.path.join(outdir, stem)
        if want_render:
            dpi = (payload.get("config") or {}).get("dpi")
            for cand in (f"render_{dpi}dpi.png", "render_native.png"):
                p = os.path.join(rdir, cand)
                if os.path.exists(p):
                    render_bgr = cv2.imread(p)
                    break
            if render_bgr is None:
                g = sorted(glob.glob(os.path.join(rdir, "render_*.png")))
                if g:
                    render_bgr = cv2.imread(g[0])
        if want_viz:
            for cand in (f"{stem}_labeled.png", f"{stem}_final.png"):
                p = os.path.join(rdir, cand)
                if os.path.exists(p):
                    with open(p, "rb") as fh:
                        viz_png = fh.read()
                    break
        return payload, render_bgr, viz_png
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _hall_predictions_render_space(render_bgr, hall_conf: float,
                                   max_edge: int) -> list:
    """Run the Roboflow hall model on a downscaled copy of the booth render, then
    scale the predictions back into the full render pixel space so hall boxes and
    booth coordinates share one coordinate system."""
    if render_bgr is None:
        return []
    h, w = render_bgr.shape[:2]
    long_edge = max(h, w)
    sf = min(1.0, float(max_edge) / float(long_edge)) if long_edge > 0 else 1.0
    if sf < 1.0:
        hall_img = cv2.resize(render_bgr,
                              (max(1, round(w * sf)), max(1, round(h * sf))),
                              interpolation=cv2.INTER_AREA)
    else:
        hall_img = render_bgr
    ok, buf = cv2.imencode(".png", hall_img)
    if not ok:
        raise RuntimeError("failed to encode hall raster")
    model = _get_rf_model("hall")
    raw = model.infer(BytesIO(buf.tobytes()), confidence=hall_conf)
    preds = serialize_model_output(raw)
    inv = (1.0 / sf) if sf > 0 else 1.0
    if inv != 1.0:
        for block in preds:
            if not isinstance(block, dict):
                continue
            for p in block.get("predictions", []):
                for k in ("x", "y", "width", "height"):
                    if isinstance(p.get(k), (int, float)):
                        p[k] = p[k] * inv
            img = block.get("image")
            if isinstance(img, dict):
                img["width"], img["height"] = w, h
    return preds


def _normalize_booths(booths: list) -> list:
    """Shape adaptive booths to the production ("PlotYouMap") booth schema so the
    existing consumer parses them unchanged. Guarantees `area` exists (derived
    from the bbox when the detector did not provide one) and exposes the PDF/OCR
    label as `name` (production replaced `id` with `name` downstream). Existing
    adaptive keys (coordinates, centroid, source, id, pdf_label, text_status, …)
    are retained — production also returned a superset."""
    out = []
    for b in booths:
        b = dict(b)
        if not isinstance(b.get("area"), (int, float)):
            bb = b.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                b["area"] = float(bb[2]) * float(bb[3])
            else:
                b["area"] = 0.0
        lab = b.get("pdf_label") or b.get("label") or ""
        lab = lab.strip() if isinstance(lab, str) else ""
        if lab:
            b["name"] = lab
        out.append(b)
    return out


def _build_hall_booth_map(hall_predictions: list, booth_detections: dict) -> dict:
    """Group booths by which hall contains their centroid (Roboflow halls are
    centre-based x/y/width/height). Booths whose centroid falls in no hall go
    under "Other". Each booth's numeric `id` is replaced by its `name` (PDF/OCR
    label) when one exists. Output shape is identical to the production endpoint.
    """
    halls = []
    for item in hall_predictions:
        if not isinstance(item, dict):
            continue
        for pred in item.get("predictions", []):
            cx, cy = pred.get("x", 0), pred.get("y", 0)
            w, h = pred.get("width", 0), pred.get("height", 0)
            if w <= 0 or h <= 0:
                continue
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            halls.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "coordinates": [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]],
            })

    booths = booth_detections.get("booths", [])

    def _label_booth(b: dict) -> dict:
        b = dict(b)
        if "name" in b:
            b["id"] = b.pop("name")
        return b

    hall_groups = [[] for _ in halls]
    other_booths = []
    for booth in booths:
        cen = booth.get("centroid")
        if not cen or len(cen) < 2:
            other_booths.append(_label_booth(booth))
            continue
        cx, cy = cen[0], cen[1]
        assigned = False
        for idx, hall in enumerate(halls):
            if hall["x1"] <= cx <= hall["x2"] and hall["y1"] <= cy <= hall["y2"]:
                hall_groups[idx].append(_label_booth(booth))
                assigned = True
                break
        if not assigned:
            other_booths.append(_label_booth(booth))

    result: dict = {}
    for i, (hall, group) in enumerate(zip(halls, hall_groups), 1):
        result[f"Hall_{i}"] = {"coordinates": hall["coordinates"], "booths": group}
    if other_booths:
        result["Other"] = {"booths": other_booths}
    return result


async def _collect_inputs(image_url, pdf_url):
    """Fetch an optional PDF (pdf_url) and an optional image (image_url) by URL.
      pdf_url     -> the PDF floor plan
      image_url   -> the IMAGE floor plan; a URL that actually serves a PDF is
                     still accepted via magic-byte sniffing
    When both resolve to usable inputs the caller uses the PDF."""
    pdf_bytes = pdf_name = img_bytes = img_name = None
    if pdf_url:
        pdf_bytes = (await fetch_image(pdf_url)).getvalue()
        pdf_name = os.path.basename(pdf_url.split("?")[0]) or "input.pdf"
    # PDF wins: only fetch image_url when no PDF was supplied, so a stray/
    # placeholder image_url (e.g. Swagger's literal "string") can't break a
    # request that already has a valid pdf_url.
    if image_url and pdf_bytes is None:
        b = (await fetch_image(image_url)).getvalue()
        nm = os.path.basename(image_url.split("?")[0]) or "input"
        if _looks_like_pdf(b):
            if pdf_bytes is None:
                pdf_bytes, pdf_name = b, nm
        else:
            img_bytes, img_name = b, nm
    return pdf_bytes, pdf_name, img_bytes, img_name


async def _hash_fields(src_bytes: bytes, is_pdf: bool, render_bgr):
    """Produce the production `hash` / `date` / `hash_status` trio. For an image
    we phash the bytes (and read any embedded date) exactly like production; for
    a PDF we phash the page render. The hash is registered for dedup tracking."""
    loop = asyncio.get_event_loop()
    img_hash = None
    img_date = None
    try:
        if not is_pdf:
            img_hash, img_date, _bytes, _w = await get_or_compute_hash(
                src_bytes, extra={"type": "Indoor"})
        elif render_bgr is not None:
            def _ph():
                rgb = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB)
                return str(imagehash.phash(Image.fromarray(rgb)))
            img_hash = await loop.run_in_executor(None, _ph)
    except Exception:
        logger.exception("hash computation failed; falling back to sha1")
    if not img_hash:
        img_hash = hashlib.sha1(src_bytes).hexdigest()
    is_present = await check_and_register_hash(img_hash)
    return img_hash, img_date, ("present" if is_present else "absent")


# ─────────────────────────────────────────────────────────────────
# DEBUG ENDPOINT  (unchanged from main)
# ─────────────────────────────────────────────────────────────────
@app.post("/debug_predict")
async def debug_predict(
    file: UploadFile = File(None),
    image_url: str = Form(None),
    _: None = Security(require_api_key),
):
    if not file and not image_url:
        raise HTTPException(status_code=400, detail="Provide file or image_url")
    try:
        if file:
            image_bytes = BytesIO(await file.read())
        else:
            image_bytes = await fetch_image(image_url)
        trail_output_raw = await run_model_async(
            _get_rf_model("trail"), BytesIO(image_bytes.getvalue()), 0.25)
        serialized = serialize_model_output(trail_output_raw)
        trail_block, preds = extract_trail_data(serialized)
        image_info = trail_block.get("image", {})
        result = {
            "raw_type": str(type(trail_output_raw)),
            "serialized_keys": list(trail_block.keys()) if trail_block else [],
            "image_info": image_info,
            "js_w": image_info.get("width"),
            "js_h": image_info.get("height"),
            "prediction_count": len(preds),
            "predictions_with_points": sum(1 for p in preds if p.get("points")),
            "first_pred_keys": list(preds[0].keys()) if preds else [],
            "first_pred_point_count": len(preds[0]["points"]) if preds and preds[0].get("points") else 0,
            "sample_point": preds[0]["points"][0] if preds and preds[0].get("points") else None,
        }
        _persist_execution("debug_predict", image_bytes.getvalue(),
                           (file.filename if file else os.path.basename(image_url.split("?")[0])), result)
        return result
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# ─────────────────────────────────────────────────────────────────
# MAIN ENDPOINT (Icons and Trails)  (unchanged from main)
# ─────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(
    file: UploadFile = File(None),
    image_url: str = Form(None),
    _: None = Security(require_api_key),
):
    if not file and not image_url:
        raise HTTPException(status_code=400, detail="Provide file or image_url")
    try:
        if file:
            image_bytes = BytesIO(await file.read())
        else:
            image_bytes = await fetch_image(image_url)
        raw_bytes = image_bytes.getvalue()

        img_hash, img_date, raw_bytes, was_written = await get_or_compute_hash(raw_bytes)
        s3_result = {"attempted": False, "reason": "no metadata change"}
        if was_written and image_url:
            s3_result = await writeback_to_s3(image_url, raw_bytes)

        is_present = await check_and_register_hash(img_hash)
        hash_status = "present" if is_present else "absent"

        if is_present and HASH_CHECK_MODE == "reject_if_present":
            raise HTTPException(status_code=409, detail={
                "message": "Duplicate image — already processed.",
                "hash": img_hash, "date": img_date, "hash_status": hash_status})
        if is_present and HASH_CHECK_MODE == "skip_if_present":
            return {"hash": img_hash, "date": img_date, "hash_status": hash_status,
                    "s3_writeback": s3_result, "icon_output": [], "trail_output": [],
                    "skipped": True}

        trail_task = run_model_async(_get_rf_model("trail"), BytesIO(raw_bytes), 0.25)
        icon_task = run_model_async(_get_rf_model("icon"), BytesIO(raw_bytes), 0.30)
        trail_output_raw, icon_output_raw = await asyncio.gather(trail_task, icon_task)

        trail_serialized = serialize_model_output(trail_output_raw)
        icon_serialized = serialize_model_output(icon_output_raw)
        trail_block, trail_predictions = extract_trail_data(trail_serialized)
        icon_block, _ = extract_trail_data(icon_serialized)

        map_image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        merged_predictions = merge_trails(predictions=trail_predictions,
                                          map_image=map_image,
                                          config=TRAIL_MERGE_CONFIG,
                                          trail_block=trail_block)
        merged_trail_block = dict(trail_block)
        merged_trail_block["predictions"] = merged_predictions

        result = {"hash": img_hash, "date": img_date, "hash_status": hash_status,
                  "s3_writeback": s3_result, "icon_output": [icon_block],
                  "trail_output": [merged_trail_block]}
        _persist_execution("predict", raw_bytes,
                           (file.filename if file else os.path.basename(image_url.split("?")[0])), result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────
# HALL + BOOTH ENDPOINT  (UPGRADED: Roboflow hall + adaptive booth pipeline)
# ─────────────────────────────────────────────────────────────────
@app.post("/hall_with_booth_predict")
async def hall_with_booth_predict(
    image_url: str = Form(None),
    pdf_url: str = Form(None),
    page: int = Form(0),
    fp_policy: str = Form(None),
    hall_conf: float = Form(0.50),
    _: None = Security(require_api_key),
):
    """Roboflow hall detection + our adaptive booth pipeline.

    Inputs (URLs):
      image_url   the IMAGE floor plan, fetched by URL
      pdf_url     a PDF floor plan,     fetched by URL
    Send either, or both. When BOTH are present the PDF is used for everything
    (booths from the PDF vector layer; halls from the PDF page render).

    Output is identical to the production endpoint:
      { hash, date, hash_status, s3_writeback,
        hall_predictions, booth_detections:{count,booths}, hall_booth_map }
    Hall boxes are scaled into the booth render's pixel space, so
    `hall_predictions`, `booth_detections.booths` and `hall_booth_map` all share
    one coordinate system.
    """
    pdf_bytes, pdf_name, img_bytes, img_name = await _collect_inputs(
        image_url, pdf_url)
    if pdf_bytes is None and img_bytes is None:
        raise HTTPException(status_code=400,
                            detail="Provide image_url and/or pdf_url")

    use_pdf = pdf_bytes is not None
    src_bytes = pdf_bytes if use_pdf else img_bytes
    src_name = pdf_name if use_pdf else img_name
    fp = fp_policy or None
    if fp is not None:
        try:
            from labeling import POLICY_CHOICES as _POLICY_CHOICES
        except Exception:
            _POLICY_CHOICES = None
        if _POLICY_CHOICES is not None and fp not in _POLICY_CHOICES:
            # Ignore junk/placeholder values (e.g. Swagger's "string").
            fp = None

    try:
        loop = asyncio.get_event_loop()
        # 1. Booth detection (long pole) — produces booths + the page render.
        payload, render_bgr, _viz = await loop.run_in_executor(
            None, _run_adaptive_booths, src_bytes, src_name, use_pdf, page, fp,
            True, False)
        # 2. Hall detection on the downscaled render, scaled back to render space.
        hall_predictions = await loop.run_in_executor(
            None, _hall_predictions_render_space, render_bgr,
            float(hall_conf), HALL_RASTER_MAX_EDGE)

        booths = _normalize_booths(payload.get("kept", []))
        booth_detections = {"count": len(booths), "booths": booths}
        hall_booth_map = _build_hall_booth_map(hall_predictions, booth_detections)

        img_hash, img_date, hash_status = await _hash_fields(
            src_bytes, use_pdf, render_bgr)

        result = {
            "hash": img_hash,
            "date": img_date,
            "hash_status": hash_status,
            "s3_writeback": {"attempted": False, "reason": "writeback disabled"},
            "hall_predictions": hall_predictions,
            "booth_detections": booth_detections,
            "hall_booth_map": hall_booth_map,
        }
        _persist_execution("hall_with_booth_predict", src_bytes, src_name, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Hall + booth prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "adaptive", "log_path": _LOG_PATH}
