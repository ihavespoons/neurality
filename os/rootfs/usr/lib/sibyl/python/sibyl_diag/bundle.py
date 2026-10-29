"""Diagnostic bundles, diagnostic IDs, and opt-in upload.

A bundle is a redacted tar.gz of one incident (or a fresh snapshot), plus the
hardware inventory and a manifest. Every bundle gets a diagnostic ID such as
SBL-7K2M-9QXD. The user quotes that ID to support, with or without an upload,
so a bundle copied off the box on a USB stick works the same way.

Nothing leaves the box unless consent mode is "manual" (the user runs
`sibyl-diag submit`) or "automatic" (incident bundles upload on their own).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config, hardware
from .redact import Redactor

# Crockford base32: no I, L, O or U, so an ID read over the phone is unambiguous.
ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
TEXT_SUFFIXES = {".txt", ".json", ".csv", ".yaml", ".log", ""}


def _check_char(body: str) -> str:
    # Odd weights are coprime with 32, so any single wrong character changes
    # the check character.
    total = sum((2 * i + 1) * ALPHABET.index(c) for i, c in enumerate(body))
    return ALPHABET[total % 32]


def new_diagnostic_id() -> str:
    body = "".join(secrets.choice(ALPHABET) for _ in range(7))
    full = body + _check_char(body)
    return f"SBL-{full[:4]}-{full[4:]}"


def valid_diagnostic_id(diag_id: str) -> bool:
    raw = diag_id.upper().removeprefix("SBL-").replace("-", "")
    if len(raw) != 8 or any(c not in ALPHABET for c in raw):
        return False
    return _check_char(raw[:7]) == raw[7]


def _add(tar: tarfile.TarFile, name: str, data: bytes, manifest: list[dict]) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = int(time.time())
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))
    manifest.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})


def build(incident_dir: Path | None, note: str = "") -> tuple[str, Path, list[dict]]:
    """Create a redacted bundle. Returns (diagnostic ID, path, manifest entries)."""
    cfg = config.load_config()
    diag_id = new_diagnostic_id()
    redactor = Redactor(config.redaction_salt())
    config.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.BUNDLE_DIR / f"{diag_id}.tar.gz"
    manifest: list[dict] = []
    budget = cfg.upload.max_bundle_bytes

    with tarfile.open(path, "w:gz") as tar:
        root = diag_id
        files = sorted(f for f in incident_dir.rglob("*") if f.is_file()) if incident_dir else []
        for f in files:
            rel = f.relative_to(incident_dir).as_posix()
            data = f.read_bytes()
            if f.suffix in TEXT_SUFFIXES:
                data = redactor.text(data.decode(errors="replace")).encode()
            else:
                # Binary files (pstore compressed records) cannot be redacted, so
                # they stay out unless they are the kernel crash text.
                continue
            if len(data) > budget:
                data = b"<truncated, kept the tail>\n" + data[-budget:]
            budget -= len(data)
            _add(tar, f"{root}/incident/{rel}", data, manifest)
        if not incident_dir:
            inv = json.dumps(hardware.inventory(), indent=2, default=str)
            _add(tar, f"{root}/hardware.json", redactor.text(inv).encode(), manifest)
        meta = {
            "diagnostic_id": diag_id,
            "device_id": config.device_id(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sibyl_version": config.sibyl_version(),
            "gpu_variant": config.gpu_variant(),
            "incident": incident_dir.name if incident_dir else None,
            "note": redactor.text(note),
            "redacted": True,
            "files": manifest,
        }
        _add(tar, f"{root}/manifest.json", json.dumps(meta, indent=2).encode(), [])
    os.chmod(path, 0o640)
    return diag_id, path, manifest


class UploadError(Exception):
    pass


def _check_endpoint(endpoint: str) -> str:
    if not endpoint:
        raise UploadError("No upload endpoint is configured. Set one with "
                          "`sibyl-diag consent manual --endpoint https://...`.")
    url = urllib.parse.urlparse(endpoint)
    local = url.hostname in ("127.0.0.1", "localhost", "::1")
    if url.scheme != "https" and not (url.scheme == "http" and local):
        raise UploadError(f"The upload endpoint must use https: {endpoint}")
    return endpoint.rstrip("/")


def endpoint() -> str:
    """The consent endpoint wins over the image default, so a fleet can override it."""
    return config.load_consent().endpoint or config.load_config().upload.endpoint


def upload(path: Path, timeout: int = 120) -> dict:
    """POST a bundle to {endpoint}/v1/bundles. Refuses when consent is off."""
    consent = config.load_consent()
    if consent.mode == "off":
        raise UploadError("Upload consent is off. Turn it on with `sibyl-diag consent manual`.")
    base = _check_endpoint(endpoint())
    data = path.read_bytes()
    diag_id = path.name.removesuffix(".tar.gz")
    req = urllib.request.Request(base + "/v1/bundles", data=data, method="POST", headers={
        "Content-Type": "application/gzip",
        "X-Sibyl-Diagnostic-Id": diag_id,
        "X-Sibyl-Device-Id": config.device_id(),
        "X-Sibyl-Version": config.sibyl_version(),
        "X-Content-SHA256": hashlib.sha256(data).hexdigest(),
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raise UploadError(f"The server refused the bundle: HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise UploadError(f"Upload failed: {exc}") from exc
    (path.parent / (path.name + ".sent")).write_text(json.dumps(body) + "\n")
    path.with_name(path.name + ".pending").unlink(missing_ok=True)
    return body


def mark_pending(path: Path) -> None:
    path.with_name(path.name + ".pending").touch()


def upload_pending() -> list[str]:
    """Retry queued bundles. Runs from a timer, and only in automatic mode."""
    if config.load_consent().mode != "automatic":
        return []
    sent = []
    for marker in sorted(config.BUNDLE_DIR.glob("*.tar.gz.pending")):
        bundle = marker.with_name(marker.name.removesuffix(".pending"))
        if not bundle.exists():
            marker.unlink()
            continue
        try:
            upload(bundle)
            sent.append(bundle.name)
        except UploadError as exc:
            print(f"sibyl-diag: {bundle.name}: {exc}", flush=True)
            break
    return sent
