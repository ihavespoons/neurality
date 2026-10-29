"""sd_notify without libsystemd."""

from __future__ import annotations

import contextlib
import os
import socket


def notify(message: str) -> None:
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock, contextlib.suppress(OSError):
        sock.sendto(message.encode(), addr)
