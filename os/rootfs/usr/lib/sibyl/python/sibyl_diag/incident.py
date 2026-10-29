"""Incident capture.

An incident is a directory under /var/lib/sibyl/diagnostics/incidents with
everything needed to debug one failure: logs, GPU state, the model that was
loaded, recent metrics and the hardware inventory. Capture runs before the
watchdog acts, because a restart or a reboot destroys most of this state.

Incidents stay on the box. Only `sibyl-diag bundle` copies one out, redacted.
"""

from __future__ import annotations

import json
import shutil
from contextlib import closing
import time
from datetime import UTC, datetime
from pathlib import Path

from . import config, hardware, host, metrics
from .llamaswap import LlamaSwap, LlamaSwapError
from .sdnotify import notify

MAX_FILE_BYTES = 20 * 1024**2


def _write(dir_: Path, name: str, content: str | bytes | object) -> None:
    path = dir_ / name
    if isinstance(content, bytes):
        path.write_bytes(content[-MAX_FILE_BYTES:])
        return
    if not isinstance(content, str):
        content = json.dumps(content, indent=2, default=str)
    if len(content) > MAX_FILE_BYTES:
        content = "<truncated, kept the tail>\n" + content[-MAX_FILE_BYTES:]
    path.write_text(content)


def _step(dir_: Path, errors: list[str], name: str, fn) -> None:
    """Run one capture step. A failed step must not stop the others."""
    # A full capture can outlast WatchdogSec of the calling service.
    notify("WATCHDOG=1")
    try:
        _write(dir_, name, fn())
    except Exception as exc:  # noqa: BLE001 - capture keeps going on any error
        errors.append(f"{name}: {type(exc).__name__}: {exc}")


def _swap_snapshot(swap: LlamaSwap) -> dict:
    out: dict = {}
    for key, fn in (("version", swap.version), ("running", swap.running),
                    ("activity", lambda: swap.activity(limit=200, order="desc"))):
        try:
            out[key] = fn()
        except LlamaSwapError as exc:
            out[key] = f"<unavailable: {exc}>"
    for entry in out["running"] if isinstance(out["running"], list) else []:
        model = entry.get("model", "")
        try:
            out.setdefault("upstream_metrics", {})[model] = {
                f"{k[0]}{{{k[1]}}}" if k[1] else k[0]: v
                for k, v in swap.upstream_metrics(model).items()}
        except LlamaSwapError as exc:
            out.setdefault("upstream_metrics", {})[model] = f"<unavailable: {exc}>"
    return out


def _gpu_tools() -> str:
    parts = []
    for cmd in (["amd-smi", "static", "--json"], ["amd-smi", "metric", "--json"],
                ["rocm-smi", "--showallinfo"], ["nvidia-smi", "-q"]):
        out = hardware.run(cmd, timeout=60)
        if out:
            parts.append(f"$ {' '.join(cmd)}\n{out}")
    return "\n\n".join(parts) or "<no GPU tool installed>"


def _pstore(dir_: Path) -> str:
    """Copy crash records that systemd-pstore archived from the last crash."""
    if not config.PSTORE_ARCHIVE.is_dir():
        return "<no pstore archive>"
    entries = sorted(config.PSTORE_ARCHIVE.iterdir(), key=lambda p: p.stat().st_mtime)
    if not entries:
        return "<pstore archive is empty>"
    latest = entries[-1]
    target = dir_ / "pstore"
    if latest.is_dir():
        shutil.copytree(latest, target, dirs_exist_ok=True)
    else:
        target.mkdir(exist_ok=True)
        shutil.copy2(latest, target)
    return f"copied {latest.name}"


def capture(reason: str, details: dict | None = None, previous_boot: bool = False) -> Path:
    cfg = config.load_config()
    now = datetime.now(UTC)
    dir_ = config.INCIDENT_DIR / f"{now:%Y%m%dT%H%M%SZ}-{reason}"
    dir_.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    swap = LlamaSwap(cfg.watchdog.llama_swap_url, timeout=cfg.watchdog.probe_timeout)

    meta = {
        "reason": reason,
        "details": details or {},
        "time": now.isoformat(),
        "boot_id": host.boot_id(),
        "device_id": config.device_id(),
        "sibyl_version": config.sibyl_version(),
        "gpu_variant": config.gpu_variant(),
        "inference_image": config.inference_backend(),
        "safe_mode": config.SAFE_MODE_FLAG.exists(),
    }
    _write(dir_, "incident.json", meta)

    if previous_boot:
        _step(dir_, errors, "journal-previous-boot.txt", lambda: host.journal("-b", "-1", "-n", "20000"))
        _step(dir_, errors, "journal-previous-boot-kernel.txt", lambda: host.journal("-k", "-b", "-1"))
        _step(dir_, errors, "pstore.txt", lambda: _pstore(dir_))
    _step(dir_, errors, "journal-kernel.txt", lambda: host.journal("-k", "-b", "0"))
    _step(dir_, errors, "journal-system.txt", lambda: host.journal("-b", "0", "--since", "-1h", "-n", "50000"))
    _step(dir_, errors, "journal-llama-swap.txt",
          lambda: host.journal("CONTAINER_NAME=llama-swap", "--since", "-2h", "-n", "50000"))
    _step(dir_, errors, "journal-sibyl-user.txt",
          lambda: host.journal(f"_UID={config.SIBYL_UID}", "--since", "-1h", "-n", "20000"))
    _step(dir_, errors, "llama-swap-logs.txt", swap.logs)
    _step(dir_, errors, "llama-swap-state.json", lambda: _swap_snapshot(swap))
    _step(dir_, errors, "llama-swap.yaml", lambda: config.LLAMA_SWAP_CONFIG.read_text())
    _step(dir_, errors, "units.txt", lambda: "\n".join([
        host.user_systemctl("status", "--no-pager", "-n", "50", "llama-swap.service",
                            "open-webui.service", "caddy.service", timeout=30).stdout,
        hardware.run(["systemctl", "--failed", "--no-pager"]),
    ]))
    _step(dir_, errors, "podman.txt", lambda: "\n".join([
        host.podman("ps", "-a", "--format", "json"),
        host.podman("images", "--digests", "--format", "json"),
    ]))
    _step(dir_, errors, "gpu-tools.txt", _gpu_tools)
    _step(dir_, errors, "hardware.json", hardware.inventory)
    _step(dir_, errors, "watchdog-state.json", lambda: config.WATCHDOG_STATE.read_text())

    def _metrics() -> str:
        with closing(metrics.connect()) as db:
            samples, requests = metrics.export_csv(db, int(time.time()) - 3600)
            _write(dir_, "requests-1h.csv", requests)
            _write(dir_, "perf-24h.json", metrics.perf_summary(db, int(time.time()) - 86400))
        return samples
    _step(dir_, errors, "samples-1h.csv", _metrics)

    if errors:
        _write(dir_, "capture-errors.txt", "\n".join(errors) + "\n")
    prune(cfg)
    return dir_


def list_incidents() -> list[Path]:
    if not config.INCIDENT_DIR.is_dir():
        return []
    return sorted((p for p in config.INCIDENT_DIR.iterdir() if p.is_dir()), key=lambda p: p.name)


def _size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def prune(cfg: config.Config) -> None:
    """Keep at most max_incidents directories and max_incident_bytes in total."""
    incidents = list_incidents()
    sizes = {p: _size(p) for p in incidents}
    total = sum(sizes.values())
    while incidents and (len(incidents) > cfg.capture.max_incidents
                         or total > cfg.capture.max_incident_bytes):
        oldest = incidents.pop(0)
        total -= sizes[oldest]
        shutil.rmtree(oldest, ignore_errors=True)
