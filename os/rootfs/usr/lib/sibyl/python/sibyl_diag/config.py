"""Paths and configuration.

Operator settings are TOML: /usr/share/sibyl/diagnostics.toml is the image
default, and /var/lib/sibyl/config/diagnostics.toml overrides it key by key.
Upload consent is separate JSON state that only `sibyl-diag consent` writes,
so a hand-edited TOML file can never turn uploads on by accident.
"""

from __future__ import annotations

import json
import os
import time
import tomllib
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# SIBYL_ROOT lets the tests point every path at a temporary tree.
ROOT = Path(os.environ.get("SIBYL_ROOT", "/"))

STATE_DIR = ROOT / "var/lib/sibyl"
CONFIG_DIR = STATE_DIR / "config"
DIAG_DIR = STATE_DIR / "diagnostics"
INCIDENT_DIR = DIAG_DIR / "incidents"
BUNDLE_DIR = DIAG_DIR / "bundles"
METRICS_DB = DIAG_DIR / "metrics.db"
WATCHDOG_STATE = DIAG_DIR / "watchdog-state.json"
# In config/, not diagnostics/: the sibyl user manager evaluates the Quadlet
# ConditionPathExists and cannot see into the root-only diagnostics directory.
SAFE_MODE_FLAG = CONFIG_DIR / "safe-mode"
RUNNING_MARKER = DIAG_DIR / "running"
DEVICE_ID_FILE = DIAG_DIR / "device-id"
REDACTION_SALT_FILE = DIAG_DIR / "redaction-salt"
CONSENT_FILE = CONFIG_DIR / "diagnostics-consent.json"
LLAMA_SWAP_CONFIG = CONFIG_DIR / "llama-swap.yaml"

DEFAULT_TOML = ROOT / "usr/share/sibyl/diagnostics.toml"
OVERRIDE_TOML = CONFIG_DIR / "diagnostics.toml"
VERSION_FILE = ROOT / "usr/share/sibyl/version"
VARIANT_FILE = ROOT / "usr/share/sibyl/gpu-variant"
BACKEND_DROPIN = ROOT / "etc/containers/systemd/users/1001/llama-swap.container.d/backend.conf"
PSTORE_ARCHIVE = ROOT / "var/lib/systemd/pstore"

SIBYL_USER = "sibyl"
SIBYL_UID = 1001

UPLOAD_MODES = ("off", "manual", "automatic")


@dataclass
class CaptureConfig:
    sample_interval: int = 10
    retention_days: int = 7
    max_incidents: int = 50
    max_incident_bytes: int = 2 * 1024**3


@dataclass
class WatchdogConfig:
    enabled: bool = True
    llama_swap_url: str = "http://127.0.0.1:18080"
    probe_interval: int = 30
    probe_timeout: int = 10
    failures_before_action: int = 3
    stall_seconds: int = 300
    max_restarts: int = 3
    restart_window: int = 900
    gpu_reset: bool = True
    reboot: bool = True
    max_reboots: int = 2
    reboot_window: int = 21600
    incident_cooldown: int = 600


@dataclass
class UploadConfig:
    endpoint: str = ""
    max_bundle_bytes: int = 50 * 1024**2


@dataclass
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)


def _apply(section: object, values: dict) -> None:
    for key, value in values.items():
        if hasattr(section, key):
            setattr(section, key, type(getattr(section, key))(value))


def load_config() -> Config:
    cfg = Config()
    for path in (DEFAULT_TOML, OVERRIDE_TOML):
        try:
            data = tomllib.loads(path.read_text())
        except FileNotFoundError:
            continue
        for name in ("capture", "watchdog", "upload"):
            _apply(getattr(cfg, name), data.get(name, {}))
    return cfg


@dataclass
class Consent:
    mode: str = "off"
    endpoint: str = ""
    changed_at: float = 0.0


def load_consent() -> Consent:
    try:
        data = json.loads(CONSENT_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return Consent()
    mode = data.get("mode", "off")
    if mode not in UPLOAD_MODES:
        mode = "off"
    return Consent(mode=mode, endpoint=str(data.get("endpoint", "")),
                   changed_at=float(data.get("changed_at", 0)))


def save_consent(mode: str, endpoint: str | None = None) -> Consent:
    if mode not in UPLOAD_MODES:
        raise ValueError(f"mode must be one of {', '.join(UPLOAD_MODES)}")
    current = load_consent()
    consent = Consent(mode=mode,
                      endpoint=current.endpoint if endpoint is None else endpoint,
                      changed_at=time.time())
    write_private(CONSENT_FILE, json.dumps(consent.__dict__, indent=2) + "\n")
    return consent


def write_private(path: Path, text: str) -> None:
    """Write a file atomically with mode 0640."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _random_file(path: Path, make: Callable[[], str]) -> str:
    try:
        value = path.read_text().strip()
        if value:
            return value
    except FileNotFoundError:
        pass
    value = make()
    write_private(path, value + "\n")
    return value


def device_id() -> str:
    """A random ID for this box. It is not derived from machine-id or serials."""
    return _random_file(DEVICE_ID_FILE, lambda: str(uuid.uuid4()))


def redaction_salt() -> bytes:
    return bytes.fromhex(_random_file(REDACTION_SALT_FILE, lambda: os.urandom(16).hex()))


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return default


def sibyl_version() -> str:
    return read_text(VERSION_FILE, "unknown")


def gpu_variant() -> str:
    return read_text(VARIANT_FILE, "unknown")


def inference_backend() -> str:
    """The llama-swap image from the backend drop-in, for example ...:1.0-hip."""
    for line in read_text(BACKEND_DROPIN).splitlines():
        if line.startswith("Image="):
            return line.split("=", 1)[1].strip()
    return "unknown"
