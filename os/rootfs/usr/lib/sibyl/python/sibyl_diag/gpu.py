"""GPU discovery and live readings.

AMD cards are read from amdgpu sysfs, which needs no extra tools. NVIDIA cards
are read with nvidia-smi when it is installed. Intel cards get PCI data only.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import ROOT

SYS = ROOT / "sys"
VENDORS = {"0x1002": "amd", "0x10de": "nvidia", "0x8086": "intel"}


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _int(path: Path) -> int | None:
    raw = _read(path)
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass
class Gpu:
    index: int
    pci: str
    vendor: str
    device_id: str
    subsystem: str
    driver: str
    drm_card: str
    vram_total: int | None = None
    vbios: str = ""
    link_speed: str = ""
    link_width: str = ""
    max_link_speed: str = ""
    max_link_width: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _hwmon(device: Path) -> Path | None:
    for hw in sorted((device / "hwmon").glob("hwmon*")):
        return hw
    return None


def discover() -> list[Gpu]:
    gpus: list[Gpu] = []
    for card in sorted((SYS / "class/drm").glob("card[0-9]*")):
        if "-" in card.name:  # connectors such as card0-DP-1
            continue
        device = card / "device"
        vendor = VENDORS.get(_read(device / "vendor"), _read(device / "vendor"))
        if not vendor:
            continue
        driver_link = device / "driver"
        driver = driver_link.resolve().name if driver_link.exists() else ""
        gpus.append(Gpu(
            index=len(gpus),
            pci=device.resolve().name,
            vendor=vendor,
            device_id=_read(device / "device"),
            subsystem=f"{_read(device / 'subsystem_vendor')}:{_read(device / 'subsystem_device')}",
            driver=driver,
            drm_card=card.name,
            vram_total=_int(device / "mem_info_vram_total"),
            vbios=_read(device / "vbios_version"),
            link_speed=_read(device / "current_link_speed"),
            link_width=_read(device / "current_link_width"),
            max_link_speed=_read(device / "max_link_speed"),
            max_link_width=_read(device / "max_link_width"),
        ))
    return gpus


def sample_amd(gpu: Gpu) -> dict[str, float]:
    """One reading of an amdgpu card. Units: percent, bytes, degC, W, RPM, MHz."""
    device = SYS / "class/drm" / gpu.drm_card / "device"
    out: dict[str, float] = {}
    for name, fname in (("busy_percent", "gpu_busy_percent"),
                        ("mem_busy_percent", "mem_busy_percent"),
                        ("vram_used_bytes", "mem_info_vram_used"),
                        ("vram_total_bytes", "mem_info_vram_total"),
                        ("gtt_used_bytes", "mem_info_gtt_used")):
        v = _int(device / fname)
        if v is not None:
            out[name] = float(v)
    hw = _hwmon(device)
    if hw:
        # amdgpu labels temp1 edge, temp2 junction, temp3 mem.
        for i in (1, 2, 3):
            label = _read(hw / f"temp{i}_label") or f"temp{i}"
            v = _int(hw / f"temp{i}_input")
            if v is not None:
                out[f"temp_{label}_c"] = v / 1000
        for fname in ("power1_average", "power1_input"):
            v = _int(hw / fname)
            if v is not None:
                out["power_w"] = v / 1e6
                break
        v = _int(hw / "power1_cap")
        if v is not None:
            out["power_cap_w"] = v / 1e6
        v = _int(hw / "fan1_input")
        if v is not None:
            out["fan_rpm"] = float(v)
        for i, name in ((1, "sclk_mhz"), (2, "mclk_mhz")):
            v = _int(hw / f"freq{i}_input")
            if v is not None:
                out[name] = v / 1e6
    return out


_NV_FIELDS = ("index,pci.bus_id,utilization.gpu,utilization.memory,memory.used,memory.total,"
              "temperature.gpu,power.draw,power.limit,fan.speed,clocks.sm,clocks.mem")


def sample_nvidia() -> dict[str, dict[str, float]]:
    """Readings keyed by lower-case PCI address, from one nvidia-smi call."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        res = subprocess.run(["nvidia-smi", f"--query-gpu={_NV_FIELDS}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True)
    except (subprocess.SubprocessError, OSError):
        return {}
    names = ("busy_percent", "mem_busy_percent", "vram_used_bytes", "vram_total_bytes",
             "temp_edge_c", "power_w", "power_cap_w", "fan_percent", "sclk_mhz", "mclk_mhz")
    out: dict[str, dict[str, float]] = {}
    for line in res.stdout.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 12:
            continue
        # nvidia-smi prints 00000000:03:00.0; sysfs uses 0000:03:00.0.
        pci = cols[1].lower()[-12:]
        reading: dict[str, float] = {}
        for name, raw in zip(names, cols[2:], strict=True):
            try:
                v = float(raw)
            except ValueError:
                continue
            reading[name] = v * 1024**2 if name.startswith("vram") else v
        out[pci] = reading
    return out


def sample_all(gpus: list[Gpu]) -> dict[int, dict[str, float]]:
    nv = sample_nvidia() if any(g.vendor == "nvidia" for g in gpus) else {}
    out: dict[int, dict[str, float]] = {}
    for g in gpus:
        if g.vendor == "amd":
            out[g.index] = sample_amd(g)
        elif g.vendor == "nvidia":
            out[g.index] = nv.get(g.pci.lower()[-12:], {})
    return out


def debugfs_recover_path(gpu: Gpu) -> Path | None:
    """The amdgpu debugfs file that forces a GPU reset when read."""
    for entry in sorted((SYS / "kernel/debug/dri").glob("*")):
        recover = entry / "amdgpu_gpu_recover"
        if recover.exists() and gpu.pci in _read(entry / "name"):
            return recover
    return None
