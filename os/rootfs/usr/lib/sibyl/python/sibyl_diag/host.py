"""Calls into systemd and Podman for the sibyl user's units."""

from __future__ import annotations

import subprocess

from .config import SIBYL_USER
from .hardware import run

USER_MACHINE = f"{SIBYL_USER}@"


def user_systemctl(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run systemctl against the sibyl user manager. Needs root."""
    return subprocess.run(["systemctl", "--user", "-M", USER_MACHINE, *args],
                          capture_output=True, text=True, timeout=timeout)


def unit_property(unit: str, prop: str) -> str:
    try:
        res = user_systemctl("show", "-p", prop, "--value", unit, timeout=15)
    except (subprocess.SubprocessError, OSError):
        return ""
    return res.stdout.strip()


def restart_llama_swap() -> str:
    res = user_systemctl("restart", "llama-swap.service", timeout=960)
    return (res.stdout + res.stderr).strip() or f"exit {res.returncode}"


def podman(*args: str, timeout: int = 60) -> str:
    return run(["runuser", "-u", SIBYL_USER, "--", "env", "XDG_RUNTIME_DIR=/run/user/1001",
                "podman", *args], timeout=timeout)


def journal(*args: str, timeout: int = 60) -> str:
    return run(["journalctl", "--no-pager", "-o", "short-iso-precise", *args], timeout=timeout)


def boot_id() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id") as fh:
            return fh.read().strip()
    except OSError:
        return ""
