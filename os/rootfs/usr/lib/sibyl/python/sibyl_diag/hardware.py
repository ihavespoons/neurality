"""Hardware and software inventory.

Serial numbers, MAC addresses and the machine ID are left out on purpose.
The inventory goes into every bundle, so it must be safe to send as it is.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess

from . import config, gpu
from .config import ROOT

SYS = ROOT / "sys"
PROC = ROOT / "proc"


def run(cmd: list[str], timeout: int = 30) -> str:
    if not shutil.which(cmd[0]):
        return ""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError) as exc:
        return f"<{cmd[0]} failed: {exc}>"
    return res.stdout + (res.stderr if res.returncode else "")


def _read(path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def cpu() -> dict:
    model, flags = "", ""
    for line in _read(PROC / "cpuinfo").splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "model name" and not model:
            model = value.strip()
        elif key == "flags" and not flags:
            flags = value.strip()
    wanted = ("avx2", "avx512f", "avx512_bf16", "avx_vnni", "amx_tile")
    return {
        "model": model,
        "logical_cpus": os.cpu_count(),
        "flags": [f for f in wanted if f in flags.split()],
    }


def memory() -> dict:
    out = {}
    for line in _read(PROC / "meminfo").splitlines():
        key, _, value = line.partition(":")
        if key in ("MemTotal", "SwapTotal", "HugePages_Total"):
            out[key] = int(value.split()[0]) * (1024 if "kB" in value else 1)
    return out


def dmi() -> dict:
    fields = ("sys_vendor", "product_name", "board_vendor", "board_name",
              "bios_vendor", "bios_version", "bios_date")
    return {f: _read(SYS / "class/dmi/id" / f) for f in fields}


def watchdogs() -> list[dict]:
    out = []
    for wd in sorted((SYS / "class/watchdog").glob("watchdog*")):
        out.append({
            "name": wd.name,
            "identity": _read(wd / "identity"),
            "timeout": _read(wd / "timeout"),
            "bootstatus": _read(wd / "bootstatus"),
        })
    return out


def disks() -> object:
    raw = run(["lsblk", "-J", "-b", "-o", "NAME,TYPE,SIZE,MODEL,ROTA,TRAN,FSTYPE,MOUNTPOINTS"])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def bootc_status() -> dict:
    raw = run(["bootc", "status", "--json"])
    try:
        status = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    # Keep the image references and digests, not the whole document.
    out = {}
    for slot in ("booted", "staged", "rollback"):
        entry = (status.get("status") or {}).get(slot) or {}
        image = entry.get("image") or {}
        out[slot] = {
            "image": (image.get("image") or {}).get("image"),
            "digest": image.get("imageDigest"),
            "version": image.get("version"),
        }
    return out


def os_release() -> dict:
    out = {}
    for line in _read(ROOT / "etc/os-release").splitlines():
        key, _, value = line.partition("=")
        if key in ("NAME", "VERSION_ID", "VARIANT_ID", "PRETTY_NAME"):
            out[key] = value.strip('"')
    return out


def inventory() -> dict:
    return {
        "sibyl": {
            "version": config.sibyl_version(),
            "gpu_variant": config.gpu_variant(),
            "inference_image": config.inference_backend(),
        },
        "os": os_release(),
        "kernel": platform.release(),
        "kernel_cmdline": _read(PROC / "cmdline"),
        "bootc": bootc_status(),
        "cpu": cpu(),
        "memory": memory(),
        "board": dmi(),
        "gpus": [g.to_dict() for g in gpu.discover()],
        "disks": disks(),
        "watchdog_devices": watchdogs(),
        "pstore_backend": _read(SYS / "module/pstore/parameters/backend"),
        "efi": (SYS / "firmware/efi").exists(),
        "secure_boot": run(["mokutil", "--sb-state"]).strip(),
    }
