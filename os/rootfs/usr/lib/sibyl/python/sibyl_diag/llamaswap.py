"""HTTP client for the llama-swap management API (v266).

llama-swap publishes only on 127.0.0.1:18080 on the host. The routes used here
are /health, /running, /logs, /api/version, /api/metrics/activity, and
/upstream/<model>/{health,metrics}, which llama-swap proxies to llama-server.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from . import prom


class LlamaSwapError(Exception):
    pass


class LlamaSwap:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, timeout: float | None = None) -> bytes:
        try:
            with urllib.request.urlopen(self.base + path, timeout=timeout or self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            exc.close()
            raise LlamaSwapError(f"GET {path}: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise LlamaSwapError(f"GET {path}: {exc}") from exc

    def _json(self, path: str) -> object:
        try:
            return json.loads(self._get(path))
        except json.JSONDecodeError as exc:
            raise LlamaSwapError(f"GET {path}: invalid JSON") from exc

    def health(self) -> bool:
        return self._get("/health").strip() == b"OK"

    def running(self) -> list[dict]:
        """Models that llama-swap has started, with their state (ready, starting, ...)."""
        data = self._json("/running")
        return list(data.get("running", [])) if isinstance(data, dict) else []

    def upstream_health(self, model: str) -> bool:
        self._get(f"/upstream/{urllib.parse.quote(model, safe='')}/health")
        return True

    def upstream_metrics(self, model: str) -> dict[tuple[str, str], float]:
        text = self._get(f"/upstream/{urllib.parse.quote(model, safe='')}/metrics").decode()
        return prom.parse(text)

    def activity(self, min_id: int = 0, limit: int = 500, order: str = "asc") -> list[dict]:
        """Per-request token stats sorted by id, from id min_id upwards."""
        query = {"limit": str(limit), "sort": "id", "order": order}
        if min_id > 0:
            query["min_id"] = str(min_id)
        data = self._json("/api/metrics/activity?" + urllib.parse.urlencode(query))
        return list(data.get("data", [])) if isinstance(data, dict) else []

    def logs(self) -> str:
        return self._get("/logs", timeout=30).decode(errors="replace")

    def version(self) -> object:
        return self._json("/api/version")
