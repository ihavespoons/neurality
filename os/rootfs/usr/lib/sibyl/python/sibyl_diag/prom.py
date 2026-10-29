"""Minimal parser for the Prometheus text format that llama-server emits."""

from __future__ import annotations

import re

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse(text: str) -> dict[tuple[str, str], float]:
    """Return {(name, labels): value}. Labels are kept as a sorted k=v string."""
    out: dict[tuple[str, str], float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, raw_labels, raw_value = m.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        labels = ""
        if raw_labels:
            labels = ",".join(f"{k}={v}" for k, v in sorted(_LABEL.findall(raw_labels)))
        out[(name, labels)] = value
    return out


def value(metrics: dict[tuple[str, str], float], name: str, default: float = 0.0) -> float:
    return metrics.get((name, ""), default)
