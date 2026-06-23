# # # import os
# # # import json
# # # import io
# # # from datetime import datetime, timezone
# # # from PIL import Image
# # # import imagehash
# # # import piexif

# # # # ─────────────────────────────────────────────────────────────────
# # # # CONFIG — persistent path
# # # # ─────────────────────────────────────────────────────────────────
# # # # Put hash_db.json OUTSIDE the deployment folder so a fresh git deploy
# # # # doesn't wipe it out. Set HASH_DB_PATH env var on your server to
# # # # something persistent, e.g. /var/data/hash_db.json on Render,
# # # # /opt/persistent/hash_db.json on a VPS, or a mounted disk path.
# # # #
# # # # Default falls back to a folder called "data/" one level above the
# # # # project, which usually survives deploys; override in production.
# # # HASH_DB_PATH = os.getenv(
# # #     "HASH_DB_PATH",
# # #     os.path.join(
# # #         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
# # #         "data",
# # #         "hash_db.json",
# # #     ),
# # # )

# # # print(HASH_DB_PATH)

# # # # Make sure parent directory exists
# # # os.makedirs(os.path.dirname(HASH_DB_PATH), exist_ok=True)


# # # # ─────────────────────────────────────────────────────────────────
# # # # LOAD / SAVE
# # # # ─────────────────────────────────────────────────────────────────
# # # def load_hash_db():
# # #     if not os.path.exists(HASH_DB_PATH):
# # #         return set()

# # #     try:
# # #         with open(HASH_DB_PATH, "r") as f:
# # #             content = f.read().strip()
# # #             if not content:
# # #                 return set()
# # #             return set(json.loads(content))
# # #     except json.JSONDecodeError:
# # #         print("WARNING: hash_db.json is corrupted. Resetting.")
# # #         return set()


# # # def save_hash_db(hash_set):
# # #     # Write atomically: write to temp file then rename, so a crash
# # #     # mid-write doesn't corrupt the DB.
# # #     tmp_path = HASH_DB_PATH + ".tmp"
# # #     with open(tmp_path, "w") as f:
# # #         json.dump(list(hash_set), f, indent=2)
# # #     os.replace(tmp_path, HASH_DB_PATH)


# # # # ─────────────────────────────────────────────────────────────────
# # # # IMAGE METADATA (EXIF) — embed hash + date
# # # # ─────────────────────────────────────────────────────────────────
# # # # We store our data inside EXIF UserComment as a small JSON blob:
# # # #   {"phash": "abc123...", "date": "2026-04-27T12:34:56Z"}

# # # _USER_COMMENT_PREFIX = b"ASCII\x00\x00\x00"  # standard EXIF UserComment charset prefix


# # # def _parse_user_comment(raw):
# # #     if not raw:
# # #         return None
# # #     payload = raw[8:] if len(raw) > 8 else raw
# # #     try:
# # #         text = payload.decode("utf-8", errors="ignore").strip("\x00").strip()
# # #         if not text:
# # #             return None
# # #         return json.loads(text)
# # #     except (json.JSONDecodeError, UnicodeDecodeError):
# # #         return None


# # # def read_image_metadata(raw_bytes: bytes):
# # #     """
# # #     Read embedded {phash, date} from image EXIF UserComment.
# # #     Returns (phash, date_str) or (None, None) if absent / unsupported format.
# # #     """
# # #     try:
# # #         exif_dict = piexif.load(raw_bytes)
# # #     except Exception:
# # #         return None, None

# # #     user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
# # #     parsed = _parse_user_comment(user_comment)
# # #     if not parsed:
# # #         return None, None

# # #     return parsed.get("phash"), parsed.get("date")


# # # def write_image_metadata(raw_bytes: bytes, phash: str, date_str: str = None):
# # #     """
# # #     Embed {phash, date} into image EXIF UserComment.
# # #     If both are ALREADY present (and phash matches), returns the original bytes
# # #     unchanged. Otherwise returns new bytes with the metadata written.

# # #     Only works reliably for JPEG. Non-JPEG returns unchanged bytes.
# # #     """
# # #     if date_str is None:
# # #         date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# # #     try:
# # #         img = Image.open(io.BytesIO(raw_bytes))
# # #         fmt = (img.format or "").upper()
# # #     except Exception:
# # #         return raw_bytes

# # #     if fmt != "JPEG":
# # #         return raw_bytes

# # #     # Already tagged with the same hash? leave it alone
# # #     existing_hash, existing_date = read_image_metadata(raw_bytes)
# # #     if existing_hash == phash and existing_date:
# # #         return raw_bytes

# # #     try:
# # #         exif_dict = piexif.load(raw_bytes)
# # #     except Exception:
# # #         exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

# # #     payload = json.dumps({"phash": phash, "date": date_str}).encode("utf-8")
# # #     exif_dict.setdefault("Exif", {})[piexif.ExifIFD.UserComment] = (
# # #         _USER_COMMENT_PREFIX + payload
# # #     )

# # #     try:
# # #         exif_bytes = piexif.dump(exif_dict)
# # #     except Exception:
# # #         return raw_bytes

# # #     out = io.BytesIO()
# # #     img.save(out, format="JPEG", exif=exif_bytes, quality=95)
# # #     return out.getvalue()

# # import os
# # import json
# # import io
# # import re
# # from datetime import datetime, timezone
# # from urllib.parse import urlparse
# # from PIL import Image
# # import imagehash
# # import piexif

# # # ─────────────────────────────────────────────────────────────────
# # # CONFIG — persistent path
# # # ─────────────────────────────────────────────────────────────────
# # HASH_DB_PATH = os.getenv(
# #     "HASH_DB_PATH",
# #     os.path.join(
# #         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
# #         "data",
# #         "hash_db.json",
# #     ),
# # )
# # os.makedirs(os.path.dirname(HASH_DB_PATH), exist_ok=True)


# # # ─────────────────────────────────────────────────────────────────
# # # LOAD / SAVE hash DB
# # # ─────────────────────────────────────────────────────────────────
# # def load_hash_db():
# #     if not os.path.exists(HASH_DB_PATH):
# #         return set()
# #     try:
# #         with open(HASH_DB_PATH, "r") as f:
# #             content = f.read().strip()
# #             if not content:
# #                 return set()
# #             return set(json.loads(content))
# #     except json.JSONDecodeError:
# #         print("WARNING: hash_db.json is corrupted. Resetting.")
# #         return set()


# # def save_hash_db(hash_set):
# #     tmp_path = HASH_DB_PATH + ".tmp"
# #     with open(tmp_path, "w") as f:
# #         json.dump(list(hash_set), f, indent=2)
# #     os.replace(tmp_path, HASH_DB_PATH)


# # # ─────────────────────────────────────────────────────────────────
# # # IMAGE METADATA (EXIF) — embed hash + date
# # # ─────────────────────────────────────────────────────────────────
# # _USER_COMMENT_PREFIX = b"ASCII\x00\x00\x00"


# # def _parse_user_comment(raw):
# #     if not raw:
# #         return None
# #     payload = raw[8:] if len(raw) > 8 else raw
# #     try:
# #         text = payload.decode("utf-8", errors="ignore").strip("\x00").strip()
# #         if not text:
# #             return None
# #         return json.loads(text)
# #     except (json.JSONDecodeError, UnicodeDecodeError):
# #         return None


# # def read_image_metadata(raw_bytes: bytes):
# #     """
# #     Read embedded {phash, date} from image EXIF UserComment.
# #     Returns (phash, date_str) or (None, None) if absent.
# #     """
# #     try:
# #         exif_dict = piexif.load(raw_bytes)
# #     except Exception:
# #         return None, None

# #     user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
# #     parsed = _parse_user_comment(user_comment)
# #     if not parsed:
# #         return None, None
# #     return parsed.get("phash"), parsed.get("date")


# # def write_image_metadata(raw_bytes: bytes, phash: str, date_str: str = None):
# #     """
# #     Embed {phash, date} into image EXIF UserComment.
# #     If both are ALREADY present (and phash matches), returns the original bytes.
# #     Otherwise returns NEW bytes with metadata written.
# #     Only works for JPEG. Non-JPEG returns unchanged bytes.
# #     """
# #     if date_str is None:
# #         date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# #     try:
# #         img = Image.open(io.BytesIO(raw_bytes))
# #         fmt = (img.format or "").upper()
# #     except Exception:
# #         return raw_bytes

# #     if fmt != "JPEG":
# #         return raw_bytes

# #     existing_hash, existing_date = read_image_metadata(raw_bytes)
# #     if existing_hash == phash and existing_date:
# #         return raw_bytes

# #     try:
# #         exif_dict = piexif.load(raw_bytes)
# #     except Exception:
# #         exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

# #     payload = json.dumps({"phash": phash, "date": date_str}).encode("utf-8")
# #     exif_dict.setdefault("Exif", {})[piexif.ExifIFD.UserComment] = (
# #         _USER_COMMENT_PREFIX + payload
# #     )

# #     try:
# #         exif_bytes = piexif.dump(exif_dict)
# #     except Exception:
# #         return raw_bytes

# #     out = io.BytesIO()
# #     img.save(out, format="JPEG", exif=exif_bytes, quality=95)
# #     return out.getvalue()


# # # ─────────────────────────────────────────────────────────────────
# # # S3 SUPPORT — parse URLs and upload back
# # # ─────────────────────────────────────────────────────────────────
# # def parse_s3_url(url: str):
# #     """
# #     Parse an S3 URL into (bucket, key, region) or None if not an S3 URL.

# #     Supports:
# #       https://<bucket>.s3.amazonaws.com/<key>
# #       https://<bucket>.s3.<region>.amazonaws.com/<key>
# #       https://s3.amazonaws.com/<bucket>/<key>
# #       https://s3.<region>.amazonaws.com/<bucket>/<key>
# #       s3://<bucket>/<key>
# #     """
# #     if not url:
# #         return None

# #     if url.startswith("s3://"):
# #         rest = url[5:]
# #         if "/" not in rest:
# #             return None
# #         bucket, key = rest.split("/", 1)
# #         return bucket, key, None

# #     parsed = urlparse(url)
# #     host = parsed.netloc
# #     path = parsed.path.lstrip("/")
# #     if not host or not path:
# #         return None

# #     # Virtual-hosted style: <bucket>.s3[.<region>].amazonaws.com
# #     m = re.match(r"^([^.]+)\.s3(?:[.-]([a-z0-9-]+))?\.amazonaws\.com$", host)
# #     if m:
# #         bucket = m.group(1)
# #         region = m.group(2)
# #         return bucket, path, region

# #     # Path style: s3[.<region>].amazonaws.com/<bucket>/<key>
# #     m = re.match(r"^s3(?:[.-]([a-z0-9-]+))?\.amazonaws\.com$", host)
# #     if m:
# #         region = m.group(1)
# #         if "/" not in path:
# #             return None
# #         bucket, key = path.split("/", 1)
# #         return bucket, key, region

# #     return None


# # def upload_bytes_to_s3(bucket: str, key: str, data: bytes,
# #                        region: str = None, content_type: str = "image/jpeg"):
# #     """
# #     Upload bytes to S3, overwriting the existing key.
# #     Requires boto3 and AWS creds (env vars / IAM role / ~/.aws/credentials).
# #     Returns True on success, False on failure.
# #     """
# #     try:
# #         import boto3
# #         from botocore.exceptions import ClientError, NoCredentialsError
# #     except ImportError:
# #         print("WARNING: boto3 not installed — cannot upload to S3. Run: pip install boto3")
# #         return False

# #     try:
# #         s3_client = boto3.client("s3", region_name=region) if region else boto3.client("s3")
# #         s3_client.put_object(
# #             Bucket=bucket,
# #             Key=key,
# #             Body=data,
# #             ContentType=content_type,
# #         )
# #         return True
# #     except (ClientError, NoCredentialsError) as e:
# #         print(f"ERROR: S3 upload failed for s3://{bucket}/{key} -> {e}")
# #         return False
# #     except Exception as e:
# #         print(f"ERROR: Unexpected S3 upload failure -> {e}")
# #         return False

# import os
# import json
# import io
# import re
# from datetime import datetime, timezone
# from urllib.parse import urlparse
# from PIL import Image, PngImagePlugin
# import imagehash
# import piexif

# # ─────────────────────────────────────────────────────────────────
# # CONFIG — persistent path
# # ─────────────────────────────────────────────────────────────────
# HASH_DB_PATH = os.getenv(
#     "HASH_DB_PATH",
#     os.path.join(
#         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#         "data",
#         "hash_db.json",
#     ),
# )
# os.makedirs(os.path.dirname(HASH_DB_PATH), exist_ok=True)


# # ─────────────────────────────────────────────────────────────────
# # CLOUDFRONT → S3 BUCKET MAPPING (only needed if you also use CF URLs)
# # ─────────────────────────────────────────────────────────────────
# def _load_cloudfront_map():
#     raw = os.getenv("CLOUDFRONT_BUCKET_MAP", "")
#     if not raw:
#         return {}
#     try:
#         return json.loads(raw)
#     except json.JSONDecodeError:
#         print("WARNING: CLOUDFRONT_BUCKET_MAP is not valid JSON. Ignoring.")
#         return {}


# CLOUDFRONT_BUCKET_MAP = _load_cloudfront_map()


# # ─────────────────────────────────────────────────────────────────
# # LOAD / SAVE hash DB
# # ─────────────────────────────────────────────────────────────────
# def load_hash_db():
#     if not os.path.exists(HASH_DB_PATH):
#         return set()
#     try:
#         with open(HASH_DB_PATH, "r") as f:
#             content = f.read().strip()
#             if not content:
#                 return set()
#             return set(json.loads(content))
#     except json.JSONDecodeError:
#         print("WARNING: hash_db.json is corrupted. Resetting.")
#         return set()


# def save_hash_db(hash_set):
#     tmp_path = HASH_DB_PATH + ".tmp"
#     with open(tmp_path, "w") as f:
#         json.dump(list(hash_set), f, indent=2)
#     os.replace(tmp_path, HASH_DB_PATH)


# # ─────────────────────────────────────────────────────────────────
# # IMAGE METADATA — supports both JPEG (EXIF) and PNG (tEXt chunks)
# # ─────────────────────────────────────────────────────────────────
# # JPEG: stored in EXIF UserComment as JSON
# # PNG:  stored as two tEXt chunks: "phash" and "date"
# #
# # Both use the SAME interface: read_image_metadata() and write_image_metadata().

# _USER_COMMENT_PREFIX = b"ASCII\x00\x00\x00"  # EXIF UserComment charset prefix

# # PNG metadata key names (these appear as text chunks in the PNG file)
# _PNG_KEY_PHASH = "phash"
# _PNG_KEY_DATE = "date"


# def _parse_user_comment(raw):
#     if not raw:
#         return None
#     payload = raw[8:] if len(raw) > 8 else raw
#     try:
#         text = payload.decode("utf-8", errors="ignore").strip("\x00").strip()
#         if not text:
#             return None
#         return json.loads(text)
#     except (json.JSONDecodeError, UnicodeDecodeError):
#         return None


# def _detect_format(raw_bytes: bytes):
#     try:
#         img = Image.open(io.BytesIO(raw_bytes))
#         return (img.format or "").upper()
#     except Exception:
#         return None


# def _read_metadata_jpeg(raw_bytes: bytes):
#     try:
#         exif_dict = piexif.load(raw_bytes)
#     except Exception:
#         return None, None

#     user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
#     parsed = _parse_user_comment(user_comment)
#     if not parsed:
#         return None, None
#     return parsed.get("phash"), parsed.get("date")


# def _read_metadata_png(raw_bytes: bytes):
#     try:
#         img = Image.open(io.BytesIO(raw_bytes))
#         # Pillow exposes tEXt chunks via img.info and img.text
#         info = getattr(img, "text", None) or img.info or {}
#         phash = info.get(_PNG_KEY_PHASH)
#         date = info.get(_PNG_KEY_DATE)
#         if phash and date:
#             return phash, date
#         return None, None
#     except Exception:
#         return None, None


# def read_image_metadata(raw_bytes: bytes):
#     """
#     Read embedded {phash, date} from image metadata.
#     Returns (phash, date_str) or (None, None) if absent.
#     Supports JPEG (via EXIF) and PNG (via tEXt chunks).
#     """
#     fmt = _detect_format(raw_bytes)
#     if fmt == "JPEG":
#         return _read_metadata_jpeg(raw_bytes)
#     if fmt == "PNG":
#         return _read_metadata_png(raw_bytes)
#     return None, None


# def _write_metadata_jpeg(raw_bytes: bytes, phash: str, date_str: str):
#     try:
#         exif_dict = piexif.load(raw_bytes)
#     except Exception:
#         exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

#     payload = json.dumps({"phash": phash, "date": date_str}).encode("utf-8")
#     exif_dict.setdefault("Exif", {})[piexif.ExifIFD.UserComment] = (
#         _USER_COMMENT_PREFIX + payload
#     )

#     try:
#         exif_bytes = piexif.dump(exif_dict)
#     except Exception:
#         return raw_bytes

#     img = Image.open(io.BytesIO(raw_bytes))
#     out = io.BytesIO()
#     img.save(out, format="JPEG", exif=exif_bytes, quality=95)
#     return out.getvalue()


# def _write_metadata_png(raw_bytes: bytes, phash: str, date_str: str):
#     try:
#         img = Image.open(io.BytesIO(raw_bytes))

#         # Build PngInfo, preserving any existing text chunks
#         png_info = PngImagePlugin.PngInfo()
#         existing = getattr(img, "text", None) or {}
#         for k, v in existing.items():
#             if k in (_PNG_KEY_PHASH, _PNG_KEY_DATE):
#                 continue  # we'll re-add these
#             try:
#                 png_info.add_text(k, v)
#             except Exception:
#                 pass

#         png_info.add_text(_PNG_KEY_PHASH, phash)
#         png_info.add_text(_PNG_KEY_DATE, date_str)

#         out = io.BytesIO()
#         img.save(out, format="PNG", pnginfo=png_info, optimize=False)
#         return out.getvalue()
#     except Exception as e:
#         print(f"WARNING: PNG metadata write failed: {e}")
#         return raw_bytes


# def write_image_metadata(raw_bytes: bytes, phash: str, date_str: str = None):
#     """
#     Embed {phash, date} into image metadata.
#     Returns original bytes if metadata is already present and matches phash.
#     Otherwise returns NEW bytes.

#     Supports JPEG (EXIF UserComment) and PNG (tEXt chunks).
#     Other formats: returns unchanged bytes.
#     """
#     if date_str is None:
#         date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

#     fmt = _detect_format(raw_bytes)
#     if fmt not in ("JPEG", "PNG"):
#         return raw_bytes

#     existing_hash, existing_date = read_image_metadata(raw_bytes)
#     if existing_hash == phash and existing_date:
#         return raw_bytes  # already tagged correctly — leave alone

#     if fmt == "JPEG":
#         return _write_metadata_jpeg(raw_bytes, phash, date_str)
#     if fmt == "PNG":
#         return _write_metadata_png(raw_bytes, phash, date_str)
#     return raw_bytes


# def content_type_for_bytes(raw_bytes: bytes) -> str:
#     """Return the appropriate Content-Type header value for these bytes."""
#     fmt = _detect_format(raw_bytes)
#     if fmt == "JPEG":
#         return "image/jpeg"
#     if fmt == "PNG":
#         return "image/png"
#     if fmt == "WEBP":
#         return "image/webp"
#     if fmt == "GIF":
#         return "image/gif"
#     return "application/octet-stream"


# # ─────────────────────────────────────────────────────────────────
# # URL PARSING — S3 + CloudFront
# # ─────────────────────────────────────────────────────────────────
# def parse_s3_url(url: str):
#     """Resolve URL to (bucket, key, region) or None."""
#     if not url:
#         return None

#     if url.startswith("s3://"):
#         rest = url[5:]
#         if "/" not in rest:
#             return None
#         bucket, key = rest.split("/", 1)
#         return bucket, key, None

#     parsed = urlparse(url)
#     host = parsed.netloc.lower()
#     path = parsed.path.lstrip("/")
#     if not host or not path:
#         return None

#     # CloudFront — look up in map
#     if host.endswith(".cloudfront.net"):
#         cfg = CLOUDFRONT_BUCKET_MAP.get(host)
#         if not cfg or not cfg.get("bucket"):
#             print(f"WARNING: CloudFront host {host!r} not in CLOUDFRONT_BUCKET_MAP")
#             return None
#         bucket = cfg["bucket"]
#         region = cfg.get("region")
#         key_prefix = cfg.get("key_prefix", "")
#         key = (key_prefix.strip("/") + "/" + path).strip("/") if key_prefix else path
#         return bucket, key, region

#     # Virtual-hosted S3: <bucket>.s3[.<region>].amazonaws.com
#     m = re.match(r"^([^.]+)\.s3(?:[.-]([a-z0-9-]+))?\.amazonaws\.com$", host)
#     if m:
#         return m.group(1), path, m.group(2)

#     # Path-style S3: s3[.<region>].amazonaws.com/<bucket>/<key>
#     m = re.match(r"^s3(?:[.-]([a-z0-9-]+))?\.amazonaws\.com$", host)
#     if m:
#         region = m.group(1)
#         if "/" not in path:
#             return None
#         bucket, key = path.split("/", 1)
#         return bucket, key, region

#     return None


# def upload_bytes_to_s3(bucket: str, key: str, data: bytes,
#                        region: str = None, content_type: str = "image/jpeg"):
#     """Upload bytes to S3, overwriting the existing key."""
#     try:
#         import boto3
#         from botocore.exceptions import ClientError, NoCredentialsError
#     except ImportError:
#         print("WARNING: boto3 not installed — cannot upload to S3. Run: pip install boto3")
#         return False

#     try:
#         s3_client = boto3.client("s3", region_name=region) if region else boto3.client("s3")
#         s3_client.put_object(
#             Bucket=bucket,
#             Key=key,
#             Body=data,
#             ContentType=content_type,
#         )
#         return True
#     except (ClientError, NoCredentialsError) as e:
#         print(f"ERROR: S3 upload failed for s3://{bucket}/{key} -> {e}")
#         return False
#     except Exception as e:
#         print(f"ERROR: Unexpected S3 upload failure -> {e}")
#         return False

import os
import json
import io
import re
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse
from PIL import Image, PngImagePlugin
import imagehash
import piexif

log = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CONFIG — persistent path
# ─────────────────────────────────────────────────────────────────
HASH_DB_PATH = os.getenv(
    "HASH_DB_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "hash_db.json",
    ),
)
os.makedirs(os.path.dirname(HASH_DB_PATH), exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# CLOUDFRONT → S3 BUCKET MAPPING (only needed if you also use CF URLs)
# ─────────────────────────────────────────────────────────────────
def _load_cloudfront_map():
    logger.debug("_load_cloudfront_map() called")
    raw = os.getenv("CLOUDFRONT_BUCKET_MAP", "")
    if not raw:
        logger.debug("_load_cloudfront_map: CLOUDFRONT_BUCKET_MAP not set, returning empty map")
        return {}
    try:
        logger.debug("_load_cloudfront_map: parsing CLOUDFRONT_BUCKET_MAP JSON")
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.exception("_load_cloudfront_map: failed to parse CLOUDFRONT_BUCKET_MAP JSON")
        log.warning("CLOUDFRONT_BUCKET_MAP is not valid JSON. Ignoring.")
        return {}


CLOUDFRONT_BUCKET_MAP = _load_cloudfront_map()


# ─────────────────────────────────────────────────────────────────
# LOAD / SAVE hash DB
# ─────────────────────────────────────────────────────────────────
def load_hash_db():
    logger.debug("load_hash_db() called path=%s", HASH_DB_PATH)
    if not os.path.exists(HASH_DB_PATH):
        logger.debug("load_hash_db: DB file does not exist, returning empty set")
        return set()
    try:
        logger.debug("load_hash_db: reading DB file")
        with open(HASH_DB_PATH, "r") as f:
            content = f.read().strip()
            if not content:
                logger.debug("load_hash_db: DB file empty, returning empty set")
                return set()
            logger.debug("load_hash_db: parsing DB content (%d chars)", len(content))
            return set(json.loads(content))
    except json.JSONDecodeError:
        logger.exception("load_hash_db: failed to parse hash_db.json")
        log.warning("hash_db.json is corrupted. Resetting.")
        return set()


def save_hash_db(hash_set):
    logger.debug("save_hash_db() called count=%d", len(hash_set))
    tmp_path = HASH_DB_PATH + ".tmp"
    logger.debug("save_hash_db: writing %d hashes to temp file %s", len(hash_set), tmp_path)
    with open(tmp_path, "w") as f:
        json.dump(list(hash_set), f, indent=2)
    logger.debug("save_hash_db: atomically replacing %s", HASH_DB_PATH)
    os.replace(tmp_path, HASH_DB_PATH)


# ─────────────────────────────────────────────────────────────────
# IMAGE METADATA — supports both JPEG (EXIF) and PNG (tEXt chunks)
# ─────────────────────────────────────────────────────────────────
# JPEG: stored in EXIF UserComment as JSON
# PNG:  stored as two tEXt chunks: "phash" and "date"
#
# Both use the SAME interface: read_image_metadata() and write_image_metadata().

_USER_COMMENT_PREFIX = b"ASCII\x00\x00\x00"  # EXIF UserComment charset prefix

# PNG metadata key names (these appear as text chunks in the PNG file)
_PNG_KEY_PHASH = "phash"
_PNG_KEY_DATE = "date"


def _parse_user_comment(raw):
    logger.debug("_parse_user_comment() called raw_len=%s", len(raw) if raw else 0)
    if not raw:
        return None
    payload = raw[8:] if len(raw) > 8 else raw
    try:
        text = payload.decode("utf-8", errors="ignore").strip("\x00").strip()
        if not text:
            logger.debug("_parse_user_comment: empty payload text")
            return None
        logger.debug("_parse_user_comment: parsing JSON payload (%d chars)", len(text))
        return json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.exception("_parse_user_comment: failed to decode/parse user comment")
        return None


def _detect_format(raw_bytes: bytes):
    logger.debug("_detect_format() called bytes_len=%s", len(raw_bytes) if raw_bytes else 0)
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        logger.debug("_detect_format: detected format=%s", img.format)
        return (img.format or "").upper()
    except Exception:
        logger.exception("_detect_format: failed to open image for format detection")
        return None


def _read_metadata_jpeg(raw_bytes: bytes):
    logger.debug("_read_metadata_jpeg() called bytes_len=%s", len(raw_bytes) if raw_bytes else 0)
    try:
        logger.debug("_read_metadata_jpeg: loading EXIF via piexif")
        exif_dict = piexif.load(raw_bytes)
    except Exception:
        logger.exception("_read_metadata_jpeg: failed to load EXIF")
        return None, None

    user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
    parsed = _parse_user_comment(user_comment)
    if not parsed:
        logger.debug("_read_metadata_jpeg: no parseable UserComment metadata")
        return None, None
    logger.debug("_read_metadata_jpeg: found phash=%s date=%s", parsed.get("phash"), parsed.get("date"))
    return parsed.get("phash"), parsed.get("date")


def _read_full_metadata_jpeg(raw_bytes: bytes) -> dict:
    logger.debug("_read_full_metadata_jpeg() called bytes_len=%s", len(raw_bytes) if raw_bytes else 0)
    try:
        logger.debug("_read_full_metadata_jpeg: loading EXIF via piexif")
        exif_dict = piexif.load(raw_bytes)
    except Exception:
        logger.exception("_read_full_metadata_jpeg: failed to load EXIF")
        return {}
    user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
    return _parse_user_comment(user_comment) or {}


def _read_metadata_png(raw_bytes: bytes):
    logger.debug("_read_metadata_png() called bytes_len=%s", len(raw_bytes) if raw_bytes else 0)
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        # Pillow exposes tEXt chunks via img.info and img.text
        info = getattr(img, "text", None) or img.info or {}
        phash = info.get(_PNG_KEY_PHASH)
        date = info.get(_PNG_KEY_DATE)
        if phash and date:
            logger.debug("_read_metadata_png: found phash=%s date=%s", phash, date)
            return phash, date
        logger.debug("_read_metadata_png: no phash/date tEXt chunks present")
        return None, None
    except Exception:
        logger.exception("_read_metadata_png: failed to read PNG metadata")
        return None, None


def _read_full_metadata_png(raw_bytes: bytes) -> dict:
    logger.debug("_read_full_metadata_png() called bytes_len=%s", len(raw_bytes) if raw_bytes else 0)
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        return dict(getattr(img, "text", None) or img.info or {})
    except Exception:
        logger.exception("_read_full_metadata_png: failed to read PNG metadata")
        return {}


def read_image_metadata(raw_bytes: bytes):
    """
    Read embedded {phash, date} from image metadata.
    Returns (phash, date_str) or (None, None) if absent.
    Supports JPEG (via EXIF) and PNG (via tEXt chunks).
    """
    logger.debug("read_image_metadata() called bytes_len=%s", len(raw_bytes) if raw_bytes else 0)
    fmt = _detect_format(raw_bytes)
    logger.debug("read_image_metadata: detected format=%s", fmt)
    if fmt == "JPEG":
        return _read_metadata_jpeg(raw_bytes)
    if fmt == "PNG":
        return _read_metadata_png(raw_bytes)
    logger.debug("read_image_metadata: unsupported format, returning (None, None)")
    return None, None


def _write_metadata_jpeg(raw_bytes: bytes, phash: str, date_str: str, extra: dict = None):
    logger.debug("_write_metadata_jpeg() called phash=%s date=%s extra_keys=%s",
                 phash, date_str, list(extra.keys()) if extra else None)
    try:
        logger.debug("_write_metadata_jpeg: loading existing EXIF via piexif")
        exif_dict = piexif.load(raw_bytes)
    except Exception:
        logger.exception("_write_metadata_jpeg: failed to load existing EXIF, starting fresh")
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    meta = {"phash": phash, "date": date_str}
    if extra:
        meta.update(extra)
    logger.debug("_write_metadata_jpeg: writing UserComment with %d fields", len(meta))
    payload = json.dumps(meta).encode("utf-8")
    exif_dict.setdefault("Exif", {})[piexif.ExifIFD.UserComment] = (
        _USER_COMMENT_PREFIX + payload
    )

    try:
        exif_bytes = piexif.dump(exif_dict)
    except Exception:
        logger.exception("_write_metadata_jpeg: failed to dump EXIF, returning original bytes")
        return raw_bytes

    logger.debug("_write_metadata_jpeg: re-encoding JPEG with new EXIF")
    img = Image.open(io.BytesIO(raw_bytes))
    out = io.BytesIO()
    img.save(out, format="JPEG", exif=exif_bytes, quality=95)
    return out.getvalue()


def _write_metadata_png(raw_bytes: bytes, phash: str, date_str: str, extra: dict = None):
    logger.debug("_write_metadata_png() called phash=%s date=%s extra_keys=%s",
                 phash, date_str, list(extra.keys()) if extra else None)
    try:
        img = Image.open(io.BytesIO(raw_bytes))

        # Build PngInfo, preserving any existing text chunks except ones we're writing
        png_info = PngImagePlugin.PngInfo()
        reserved = {_PNG_KEY_PHASH, _PNG_KEY_DATE}
        if extra:
            reserved.update(extra.keys())
        existing = getattr(img, "text", None) or {}
        logger.debug("_write_metadata_png: preserving up to %d existing text chunks", len(existing))
        for k, v in existing.items():
            if k in reserved:
                continue
            try:
                png_info.add_text(k, v)
            except Exception:
                logger.exception("_write_metadata_png: failed to preserve text chunk %r", k)
                pass

        logger.debug("_write_metadata_png: adding phash/date tEXt chunks")
        png_info.add_text(_PNG_KEY_PHASH, phash)
        png_info.add_text(_PNG_KEY_DATE, date_str)
        if extra:
            logger.debug("_write_metadata_png: adding %d extra tEXt chunks", len(extra))
            for k, v in extra.items():
                png_info.add_text(k, str(v))

        logger.debug("_write_metadata_png: re-encoding PNG with new metadata")
        out = io.BytesIO()
        img.save(out, format="PNG", pnginfo=png_info, optimize=False)
        return out.getvalue()
    except Exception as e:
        logger.exception("_write_metadata_png: PNG metadata write failed")
        log.warning("PNG metadata write failed: %s", e)
        return raw_bytes


def write_image_metadata(raw_bytes: bytes, phash: str, date_str: str = None, extra: dict = None):
    """
    Embed {phash, date, ...extra} into image metadata.
    Returns original bytes if all fields are already present and match.
    Otherwise returns NEW bytes.

    Supports JPEG (EXIF UserComment) and PNG (tEXt chunks).
    Other formats: returns unchanged bytes.
    """
    logger.debug("write_image_metadata() called phash=%s date=%s extra_keys=%s",
                 phash, date_str, list(extra.keys()) if extra else None)
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.debug("write_image_metadata: defaulted date_str=%s", date_str)

    fmt = _detect_format(raw_bytes)
    logger.debug("write_image_metadata: detected format=%s", fmt)
    if fmt not in ("JPEG", "PNG"):
        logger.debug("write_image_metadata: unsupported format, returning original bytes")
        return raw_bytes

    logger.debug("write_image_metadata: reading existing metadata to compare")
    existing = _read_full_metadata_jpeg(raw_bytes) if fmt == "JPEG" else _read_full_metadata_png(raw_bytes)
    if existing.get("phash") == phash and existing.get("date"):
        if not extra or all(existing.get(k) == str(v) for k, v in extra.items()):
            logger.debug("write_image_metadata: metadata already current, leaving bytes unchanged")
            return raw_bytes  # already tagged correctly — leave alone

    if fmt == "JPEG":
        logger.debug("write_image_metadata: writing JPEG metadata")
        return _write_metadata_jpeg(raw_bytes, phash, date_str, extra)
    if fmt == "PNG":
        logger.debug("write_image_metadata: writing PNG metadata")
        return _write_metadata_png(raw_bytes, phash, date_str, extra)
    return raw_bytes


def content_type_for_bytes(raw_bytes: bytes) -> str:
    """Return the appropriate Content-Type header value for these bytes."""
    logger.debug("content_type_for_bytes() called bytes_len=%s", len(raw_bytes) if raw_bytes else 0)
    fmt = _detect_format(raw_bytes)
    logger.debug("content_type_for_bytes: detected format=%s", fmt)
    if fmt == "JPEG":
        return "image/jpeg"
    if fmt == "PNG":
        return "image/png"
    if fmt == "WEBP":
        return "image/webp"
    if fmt == "GIF":
        return "image/gif"
    return "application/octet-stream"


# ─────────────────────────────────────────────────────────────────
# URL PARSING — S3 + CloudFront
# ─────────────────────────────────────────────────────────────────
def parse_s3_url(url: str):
    """Resolve URL to (bucket, key, region) or None."""
    logger.debug("parse_s3_url() called url=%s", url)
    if not url:
        return None

    if url.startswith("s3://"):
        logger.debug("parse_s3_url: parsing s3:// scheme URL")
        rest = url[5:]
        if "/" not in rest:
            logger.debug("parse_s3_url: s3:// URL has no key separator")
            return None
        bucket, key = rest.split("/", 1)
        logger.debug("parse_s3_url: resolved s3:// bucket=%s key=%s", bucket, key)
        return bucket, key, None

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lstrip("/")
    if not host or not path:
        logger.debug("parse_s3_url: missing host or path, returning None")
        return None

    # CloudFront — look up in map
    if host.endswith(".cloudfront.net"):
        logger.debug("parse_s3_url: resolving CloudFront host=%s via bucket map", host)
        cfg = CLOUDFRONT_BUCKET_MAP.get(host)
        if not cfg or not cfg.get("bucket"):
            log.warning("CloudFront host %r not in CLOUDFRONT_BUCKET_MAP", host)
            return None
        bucket = cfg["bucket"]
        region = cfg.get("region")
        key_prefix = cfg.get("key_prefix", "")
        key = (key_prefix.strip("/") + "/" + path).strip("/") if key_prefix else path
        logger.debug("parse_s3_url: resolved CloudFront bucket=%s key=%s region=%s", bucket, key, region)
        return bucket, key, region

    # Virtual-hosted S3: <bucket>.s3[.<region>].amazonaws.com
    m = re.match(r"^([^.]+)\.s3(?:[.-]([a-z0-9-]+))?\.amazonaws\.com$", host)
    if m:
        logger.debug("parse_s3_url: matched virtual-hosted S3 bucket=%s region=%s", m.group(1), m.group(2))
        return m.group(1), path, m.group(2)

    # Path-style S3: s3[.<region>].amazonaws.com/<bucket>/<key>
    m = re.match(r"^s3(?:[.-]([a-z0-9-]+))?\.amazonaws\.com$", host)
    if m:
        region = m.group(1)
        if "/" not in path:
            logger.debug("parse_s3_url: path-style S3 URL has no key separator")
            return None
        bucket, key = path.split("/", 1)
        logger.debug("parse_s3_url: matched path-style S3 bucket=%s key=%s region=%s", bucket, key, region)
        return bucket, key, region

    logger.debug("parse_s3_url: no S3/CloudFront pattern matched, returning None")
    return None


def upload_bytes_to_s3(bucket: str, key: str, data: bytes,
                       region: str = None, content_type: str = "image/jpeg"):
    """
    Upload bytes to S3, overwriting the existing key.
    Returns dict: {"success": bool, "error": str or None, "error_code": str or None}
    """
    logger.debug("upload_bytes_to_s3() bucket=%s key=%s region=%s content_type=%s data_len=%s",
                 bucket, key, region, content_type, len(data) if data else 0)
    try:
        logger.debug("upload_bytes_to_s3: importing boto3")
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        logger.exception("upload_bytes_to_s3: boto3 import failed")
        msg = "boto3 not installed — run: pip install boto3"
        log.warning(msg)
        return {"success": False, "error": msg, "error_code": "BOTO3_NOT_INSTALLED"}

    try:
        logger.debug("upload_bytes_to_s3: creating S3 client (region=%s)", region)
        s3_client = boto3.client("s3", region_name=region) if region else boto3.client("s3")
        logger.debug("upload_bytes_to_s3: putting object s3://%s/%s", bucket, key)
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        logger.info("upload_bytes_to_s3: uploaded %s bytes to s3://%s/%s",
                    len(data) if data else 0, bucket, key)
        return {"success": True, "error": None, "error_code": None}

    except NoCredentialsError as e:
        logger.exception("upload_bytes_to_s3: AWS credentials not found")
        msg = (
            "AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "environment variables, or attach an IAM role to the server."
        )
        log.error(msg)
        return {"success": False, "error": msg, "error_code": "NO_CREDENTIALS"}

    except ClientError as e:
        logger.exception("upload_bytes_to_s3: S3 ClientError for s3://%s/%s", bucket, key)
        code = e.response.get("Error", {}).get("Code", "Unknown")
        message = e.response.get("Error", {}).get("Message", str(e))
        full = f"S3 ClientError [{code}] for s3://{bucket}/{key}: {message}"
        log.error(full)
        return {"success": False, "error": full, "error_code": code}

    except Exception as e:
        logger.exception("upload_bytes_to_s3: unexpected S3 upload failure for s3://%s/%s", bucket, key)
        msg = f"Unexpected S3 upload failure: {type(e).__name__}: {e}"
        log.error(msg)
        return {"success": False, "error": msg, "error_code": "UNEXPECTED"}
