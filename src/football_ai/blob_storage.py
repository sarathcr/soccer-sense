"""
blob_storage.py
~~~~~~~~~~~~~~~
Vercel Blob Storage helpers for persisting the trained soccer_sense model.

On Vercel (serverless), /tmp is ephemeral and wiped on every cold start.
This module uploads the trained .pkl to Vercel Blob after every training run,
and downloads it back at predictor startup so every cold start gets the
latest user-retrained model instead of falling back to the bundled default.

Environment variable required (set in Vercel Dashboard → Project → Settings → Env Vars):
    BLOB_READ_WRITE_TOKEN=vercel_blob_rw_...

No extra pip packages needed — uses only stdlib urllib, json, re.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOB_API_URL = "https://blob.vercel-storage.com"

# Consistent pathname used for every upload so the same URL is always
# overwritten (Vercel Blob replaces on same pathname by default).
MODEL_BLOB_PATHNAME = "soccer_sense_model.pkl"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_token() -> str:
    """Return the Vercel Blob read-write token from the environment."""
    return os.environ.get("BLOB_READ_WRITE_TOKEN", "")


def _get_public_url() -> str | None:
    """
    Construct the public CDN URL for the stored model blob.

    Token format:  vercel_blob_rw_{STORE_ID}_{SECRET}
    Public URL:    https://{STORE_ID}.public.blob.vercel-storage.com/{pathname}
    """
    token = _get_token()
    if not token:
        return None
    match = re.match(r"vercel_blob_rw_([A-Za-z0-9]+)_", token)
    if not match:
        return None
    store_id = match.group(1)
    return f"https://{store_id}.public.blob.vercel-storage.com/{MODEL_BLOB_PATHNAME}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_blob_enabled() -> bool:
    """Return True when BLOB_READ_WRITE_TOKEN is configured."""
    return bool(_get_token())


def upload_model(model_path: Path) -> dict | None:
    """
    Upload the trained model .pkl file to Vercel Blob.

    Overwrites the same pathname on every call so only the latest model
    is stored.  Returns the Vercel Blob response dict on success, or None
    if blob is not configured / upload fails.
    """
    token = _get_token()
    if not token:
        print("[Blob] BLOB_READ_WRITE_TOKEN not set — skipping blob upload.")
        return None

    if not model_path.exists():
        print(f"[Blob] Model file not found at {model_path} — skipping upload.")
        return None

    try:
        with open(model_path, "rb") as f:
            data = f.read()

        upload_url = f"{BLOB_API_URL}/{MODEL_BLOB_PATHNAME}"
        req = urllib.request.Request(upload_url, data=data, method="PUT")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", "application/octet-stream")
        # x-content-type tells Vercel Blob how to serve the file
        req.add_header("x-content-type", "application/octet-stream")
        # No public caching so every download always gets the latest version
        req.add_header("cache-control", "public, max-age=0, must-revalidate")

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        size_kb = len(data) / 1024
        print(f"[Blob] ✅ Model uploaded ({size_kb:.1f} KB) → {result.get('url')}")
        return result

    except Exception as exc:
        print(f"[Blob] ❌ Upload failed: {exc}")
        return None


def download_model(dest_path: Path) -> bool:
    """
    Download the latest trained model from Vercel Blob to dest_path.

    Returns True on success, False if blob is not configured, the model
    hasn't been uploaded yet (404), or any other error occurs.
    The caller should fall back to the bundled default .pkl on False.
    """
    public_url = _get_public_url()
    if not public_url:
        if _get_token():
            print("[Blob] Could not parse store ID from token — skipping download.")
        return False

    try:
        print(f"[Blob] Downloading model from {public_url} …")
        req = urllib.request.Request(public_url)
        # Pass token so private blobs also work (no harm for public blobs)
        req.add_header("Authorization", f"Bearer {_get_token()}")

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()

        if not data:
            print("[Blob] Downloaded empty response — using bundled default.")
            return False

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(data)

        size_kb = len(data) / 1024
        print(f"[Blob] ✅ Model downloaded ({size_kb:.1f} KB) → {dest_path}")
        return True

    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print("[Blob] No custom model in blob yet — using bundled default.")
        else:
            print(f"[Blob] HTTP {exc.code} downloading model: {exc.reason}")
        return False

    except Exception as exc:
        print(f"[Blob] ❌ Download failed: {exc}")
        return False


def delete_model() -> bool:
    """
    Delete the stored model blob (called on /reset so the bundled default
    takes over on the next cold start).

    Returns True on success, False otherwise.
    """
    token = _get_token()
    if not token:
        return False

    public_url = _get_public_url()
    if not public_url:
        return False

    try:
        payload = json.dumps({"urls": [public_url]}).encode("utf-8")
        req = urllib.request.Request(BLOB_API_URL, data=payload, method="DELETE")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=30):
            pass

        print("[Blob] ✅ Custom model deleted from blob — bundled default will be used.")
        return True

    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return True  # Already gone, that's fine
        print(f"[Blob] HTTP {exc.code} deleting blob: {exc.reason}")
        return False

    except Exception as exc:
        print(f"[Blob] ❌ Delete failed: {exc}")
        return False


def get_blob_status() -> dict:
    """
    Return a status dict describing the current blob configuration and whether
    a custom model is available.  Used by the /health endpoint.
    """
    token = _get_token()
    if not token:
        return {
            "enabled": False,
            "reason": "BLOB_READ_WRITE_TOKEN env var not configured",
        }

    public_url = _get_public_url()
    if not public_url:
        return {
            "enabled": True,
            "model_available": False,
            "reason": "Could not parse store ID from token",
        }

    # Use the Vercel Blob list API to check metadata without downloading
    try:
        token_val = token
        list_url = (
            f"{BLOB_API_URL}"
            f"?prefix={MODEL_BLOB_PATHNAME}&limit=1"
        )
        req = urllib.request.Request(list_url)
        req.add_header("Authorization", f"Bearer {token_val}")

        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        blobs = data.get("blobs", [])
        if not blobs:
            return {
                "enabled": True,
                "model_available": False,
                "blob_url": public_url,
                "reason": "No custom model uploaded yet",
            }

        blob = blobs[0]
        return {
            "enabled": True,
            "model_available": True,
            "blob_url": blob.get("url", public_url),
            "size_bytes": blob.get("size"),
            "uploaded_at": blob.get("uploadedAt"),
        }

    except Exception as exc:
        return {
            "enabled": True,
            "model_available": None,
            "blob_url": public_url,
            "reason": f"Status check failed: {exc}",
        }
