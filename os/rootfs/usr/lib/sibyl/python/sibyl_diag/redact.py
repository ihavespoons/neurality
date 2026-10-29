"""Redaction for bundles that leave the box.

Secrets are replaced outright. Addresses and the hostname are replaced with a
keyed hash, so the same address gives the same token in every file of one
device, but the token cannot be turned back into the address.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import socket

# Order matters: PEM blocks and bearer tokens go before the generic patterns.
SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
     "<private-key>"),
    (re.compile(r"(?i)(authorization:\s*bearer\s+)\S+"), r"\1<token>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer <token>"),
    (re.compile(r"\bnd-[0-9a-f]{32,}\b"), "<api-key>"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "<hf-token>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "<api-key>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "<github-token>"),
    (re.compile(r"(?i)\b((?:api[_-]?key|secret|password|passwd|token)\s*[=:]\s*)[\"']?[^\s\"',]+"),
     r"\1<redacted>"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<email>"),
]

IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
MAC = re.compile(r"\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b")
IPV6 = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}\b")

# Addresses that say nothing about the network the box sits on.
KEEP_IPV4 = re.compile(r"^(127\.|0\.0\.0\.0$|255\.255\.255\.255$|10\.88\.|10\.89\.)")


class Redactor:
    def __init__(self, salt: bytes, hostname: str | None = None):
        self.salt = salt
        self.hostname = hostname if hostname is not None else socket.gethostname()

    def _token(self, kind: str, value: str) -> str:
        digest = hmac.new(self.salt, value.encode(), hashlib.sha256).hexdigest()[:8]
        return f"<{kind}-{digest}>"

    def _ipv4(self, m: re.Match) -> str:
        ip = m.group(0)
        return ip if KEEP_IPV4.match(ip) else self._token("ip", ip)

    def text(self, s: str) -> str:
        for pattern, repl in SECRET_PATTERNS:
            s = pattern.sub(repl, s)
        s = MAC.sub(lambda m: self._token("mac", m.group(0).lower()), s)
        s = IPV6.sub(lambda m: self._token("ip6", m.group(0).lower()), s)
        s = IPV4.sub(self._ipv4, s)
        if self.hostname and self.hostname not in ("localhost", "sibyl"):
            s = re.sub(rf"\b{re.escape(self.hostname)}\b", "<hostname>", s)
        return s
