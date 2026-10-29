"""Metrics sampler and store.

Every sample_interval seconds the sampler records host load, GPU readings and
the llama-server counters of each loaded model. It also copies the per-request
token stats from llama-swap into the `requests` table. All of it stays in
SQLite on the box for retention_days.
"""

from __future__ import annotations

import csv
import io
import os
import sqlite3
import statistics
import time
from pathlib import Path

from . import config, gpu, prom
from .llamaswap import LlamaSwap, LlamaSwapError

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts INTEGER NOT NULL,
    metric TEXT NOT NULL,
    labels TEXT NOT NULL DEFAULT '',
    value REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS samples_metric_ts ON samples (metric, ts);
CREATE INDEX IF NOT EXISTS samples_ts ON samples (ts);
CREATE TABLE IF NOT EXISTS requests (
    ts INTEGER NOT NULL,
    model TEXT NOT NULL,
    req_path TEXT,
    status INTEGER,
    input_tokens INTEGER,
    cached_tokens INTEGER,
    output_tokens INTEGER,
    prompt_tps REAL,
    gen_tps REAL,
    duration_ms INTEGER,
    error TEXT,
    swap_id INTEGER
);
CREATE INDEX IF NOT EXISTS requests_ts ON requests (ts);
"""

# llama-server counters that give throughput when two readings are compared.
GEN_TOKENS = "llamacpp:tokens_predicted_total"
GEN_SECONDS = "llamacpp:tokens_predicted_seconds_total"
PROMPT_TOKENS = "llamacpp:prompt_tokens_total"
PROMPT_SECONDS = "llamacpp:prompt_seconds_total"


def connect(path: Path = config.METRICS_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db


def host_sample() -> dict[str, float]:
    out: dict[str, float] = {}
    load1, load5, _ = os.getloadavg()
    out["load1"], out["load5"] = load1, load5
    mem = {}
    try:
        for line in (config.ROOT / "proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            mem[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    if "MemTotal" in mem:
        out["mem_total_bytes"] = mem["MemTotal"]
        out["mem_available_bytes"] = mem.get("MemAvailable", 0)
        out["swap_used_bytes"] = mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)
    try:
        st = os.statvfs(config.STATE_DIR)
        out["state_free_bytes"] = st.f_bavail * st.f_frsize
    except OSError:
        pass
    return out


def throughput(prev: dict[tuple[str, str], float], cur: dict[tuple[str, str], float]) -> dict[str, float]:
    """Tokens per second between two llama-server metric readings.

    The counters cover busy time only, so the rate is the speed while
    generating, not tokens divided by wall-clock time.
    """
    out: dict[str, float] = {}
    for name, tok, sec in (("gen_tps", GEN_TOKENS, GEN_SECONDS),
                           ("prompt_tps", PROMPT_TOKENS, PROMPT_SECONDS)):
        dtok = prom.value(cur, tok) - prom.value(prev, tok)
        dsec = prom.value(cur, sec) - prom.value(prev, sec)
        if dtok > 0 and dsec > 0:
            out[name] = dtok / dsec
    return out


class Sampler:
    def __init__(self, cfg: config.Config, db: sqlite3.Connection):
        self.cfg = cfg
        self.db = db
        self.swap = LlamaSwap(cfg.watchdog.llama_swap_url, timeout=5)
        self.gpus = gpu.discover()
        self.prev_upstream: dict[str, dict[tuple[str, str], float]] = {}
        self.last_swap_id = self._last_swap_id()
        self.last_prune = 0.0

    def _last_swap_id(self) -> int:
        row = self.db.execute("SELECT MAX(swap_id) FROM requests").fetchone()
        return int(row[0] or 0)

    def _rows(self, ts: int, metrics: dict[str, float], labels: str = "") -> list[tuple]:
        return [(ts, k, labels, float(v)) for k, v in metrics.items()]

    def sample_once(self) -> None:
        ts = int(time.time())
        rows = self._rows(ts, {f"host_{k}": v for k, v in host_sample().items()})
        for idx, reading in gpu.sample_all(self.gpus).items():
            rows += self._rows(ts, {f"gpu_{k}": v for k, v in reading.items()}, f"gpu={idx}")
        rows += self._sample_inference(ts)
        with self.db:
            self.db.executemany("INSERT INTO samples VALUES (?, ?, ?, ?)", rows)
        self._copy_activity()
        if time.time() - self.last_prune > 3600:
            self.prune()

    def _sample_inference(self, ts: int) -> list[tuple]:
        rows: list[tuple] = []
        try:
            running = self.swap.running()
        except LlamaSwapError:
            return [(ts, "swap_up", "", 0.0)]
        rows.append((ts, "swap_up", "", 1.0))
        rows.append((ts, "swap_models_running", "", float(len(running))))
        seen = set()
        for entry in running:
            model = str(entry.get("model", ""))
            if not model or entry.get("state") != "ready":
                continue
            seen.add(model)
            try:
                cur = self.swap.upstream_metrics(model)
            except LlamaSwapError:
                rows.append((ts, "model_metrics_up", f"model={model}", 0.0))
                continue
            labels = f"model={model}"
            rows.append((ts, "model_metrics_up", labels, 1.0))
            for (name, extra), v in cur.items():
                if name.startswith("llamacpp:"):
                    rows.append((ts, name, labels + (f",{extra}" if extra else ""), v))
            prev = self.prev_upstream.get(model)
            if prev is not None:
                rows += self._rows(ts, throughput(prev, cur), labels)
            self.prev_upstream[model] = cur
        for model in set(self.prev_upstream) - seen:
            del self.prev_upstream[model]
        return rows

    def _copy_activity(self) -> None:
        try:
            entries = self.swap.activity(min_id=self.last_swap_id + 1)
        except LlamaSwapError:
            return
        if not entries:
            # llama-swap IDs start again at 1 after a restart. If the newest
            # ID is below our high-water mark, start counting from zero.
            try:
                newest = self.swap.activity(limit=1, order="desc")
            except LlamaSwapError:
                return
            if newest and int(newest[0].get("id", 0)) < self.last_swap_id:
                self.last_swap_id = 0
            return
        rows = []
        for e in entries:
            tok = e.get("tokens") or {}
            rows.append((
                _parse_ts(e.get("timestamp")), str(e.get("model", "")), e.get("req_path"),
                e.get("resp_status_code"), tok.get("input_tokens"), tok.get("cache_tokens"),
                tok.get("output_tokens"), tok.get("prompt_per_second"), tok.get("tokens_per_second"),
                e.get("duration_ms"), e.get("error_msg") or None, int(e.get("id", 0)),
            ))
        with self.db:
            self.db.executemany("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.last_swap_id = max(r[-1] for r in rows)

    def prune(self) -> None:
        cutoff = int(time.time()) - self.cfg.capture.retention_days * 86400
        with self.db:
            self.db.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
            self.db.execute("DELETE FROM requests WHERE ts < ?", (cutoff,))
        self.last_prune = time.time()


def _parse_ts(raw: object) -> int:
    from datetime import datetime
    if isinstance(raw, str):
        try:
            return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
        except ValueError:
            pass
    return int(time.time())


def run_daemon(cfg: config.Config) -> None:
    from .sdnotify import notify
    db = connect()
    sampler = Sampler(cfg, db)
    notify("READY=1")
    while True:
        started = time.monotonic()
        try:
            sampler.sample_once()
        except sqlite3.Error as exc:
            print(f"sibyl-metrics: database error: {exc}", flush=True)
        notify("WATCHDOG=1")
        time.sleep(max(1.0, cfg.capture.sample_interval - (time.monotonic() - started)))


def _pct(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def perf_summary(db: sqlite3.Connection, since: int, model: str | None = None) -> list[dict]:
    """Per-model request counts and token speed percentiles since a timestamp."""
    sql = ("SELECT model, gen_tps, prompt_tps, output_tokens, input_tokens, status, error "
           "FROM requests WHERE ts >= ?")
    args: list = [since]
    if model:
        sql += " AND model = ?"
        args.append(model)
    by_model: dict[str, list[tuple]] = {}
    for row in db.execute(sql, args):
        by_model.setdefault(row[0], []).append(row)
    out = []
    for name, rows in sorted(by_model.items()):
        gen = [r[1] for r in rows if r[1]]
        pp = [r[2] for r in rows if r[2]]
        errors = sum(1 for r in rows if (r[5] or 200) >= 400 or r[6])
        out.append({
            "model": name,
            "requests": len(rows),
            "errors": errors,
            "output_tokens": sum(r[3] or 0 for r in rows),
            "input_tokens": sum(r[4] or 0 for r in rows),
            "gen_tps_p50": round(_pct(gen, 50), 2) if gen else None,
            "gen_tps_p5": round(_pct(gen, 5), 2) if gen else None,
            "prompt_tps_p50": round(_pct(pp, 50), 2) if pp else None,
        })
    return out


def export_csv(db: sqlite3.Connection, since: int) -> tuple[str, str]:
    """Samples and requests since a timestamp, as two CSV documents."""
    out = []
    for table in ("samples", "requests"):
        buf = io.StringIO()
        cur = db.execute(f"SELECT * FROM {table} WHERE ts >= ? ORDER BY ts", (since,))
        writer = csv.writer(buf)
        writer.writerow([d[0] for d in cur.description])
        writer.writerows(cur)
        out.append(buf.getvalue())
    return out[0], out[1]
