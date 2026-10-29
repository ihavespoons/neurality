"""Unit tests for sibyl-diag. Standard library only.

Run: python3 -m unittest discover -s tests/diag -v

SIBYL_ROOT points every path at a temporary tree with a fake amdgpu sysfs,
and a local HTTP server stands in for llama-swap and the upload endpoint.
"""

from __future__ import annotations

import http.server
import json
import os
import sys
import tarfile
import tempfile
import threading
import unittest
import urllib.parse
from pathlib import Path

ROOT = Path(tempfile.mkdtemp(prefix="sibyl-diag-test-"))
os.environ["SIBYL_ROOT"] = str(ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "os/rootfs/usr/lib/sibyl/python"))

from sibyl_diag import bundle, config, gpu, incident, metrics, prom, watchdog  # noqa: E402
from sibyl_diag.redact import Redactor  # noqa: E402

METRICS_TEXT = """# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total {prompt_tokens}
llamacpp:prompt_seconds_total {prompt_seconds}
llamacpp:tokens_predicted_total {gen_tokens}
llamacpp:tokens_predicted_seconds_total {gen_seconds}
llamacpp:n_decode_total {decode}
llamacpp:requests_processing {processing}
llamacpp:requests_deferred 0
"""


class FakeSwap:
    """State served by the fake llama-swap."""

    def __init__(self):
        self.healthy = True
        self.models = {"qwen": "ready"}
        self.counters = dict(prompt_tokens=0, prompt_seconds=0, gen_tokens=0,
                             gen_seconds=0, decode=0, processing=0)
        self.activity: list[dict] = []
        self.uploads: list[tuple[dict, bytes]] = []


FAKE = FakeSwap()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes | str | object, ctype: str = "application/json"):
        if not isinstance(body, (bytes, str)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        if url.path == "/health":
            return self._send(200 if FAKE.healthy else 503, "OK" if FAKE.healthy else "down", "text/plain")
        if not FAKE.healthy:
            return self._send(503, "down", "text/plain")
        if url.path == "/running":
            return self._send(200, {"running": [{"model": m, "state": s} for m, s in FAKE.models.items()]})
        if url.path.startswith("/upstream/qwen/health"):
            return self._send(200, {"status": "ok"})
        if url.path.startswith("/upstream/qwen/metrics"):
            return self._send(200, METRICS_TEXT.format(**FAKE.counters), "text/plain")
        if url.path == "/api/metrics/activity":
            rows = sorted(FAKE.activity, key=lambda r: r["id"], reverse=q.get("order") == ["desc"])
            min_id = int(q.get("min_id", ["0"])[0])
            rows = [r for r in rows if r["id"] >= min_id][: int(q.get("limit", ["25"])[0])]
            return self._send(200, {"data": rows, "page": 1, "limit": 25, "total": len(rows)})
        if url.path == "/logs":
            return self._send(200, "srv  log line Authorization: Bearer nd-" + "a" * 48, "text/plain")
        if url.path == "/api/version":
            return self._send(200, {"version": "266"})
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        FAKE.uploads.append((dict(self.headers), body))
        self._send(201, {"diagnostic_id": self.headers["X-Sibyl-Diagnostic-Id"]})


SERVER = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=SERVER.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{SERVER.server_address[1]}"


def make_sysfs():
    dev = ROOT / "sys/devices/pci0000:00/0000:03:00.0"
    hw = dev / "hwmon/hwmon3"
    hw.mkdir(parents=True, exist_ok=True)
    files = {
        dev / "vendor": "0x1002", dev / "device": "0x744c", dev / "subsystem_vendor": "0x1002",
        dev / "subsystem_device": "0x0e3b", dev / "mem_info_vram_total": str(24 * 1024**3),
        dev / "mem_info_vram_used": str(20 * 1024**3), dev / "gpu_busy_percent": "97",
        dev / "current_link_speed": "16.0 GT/s PCIe", dev / "current_link_width": "16",
        dev / "vbios_version": "113-D7020100-102",
        hw / "temp1_label": "edge", hw / "temp1_input": "65000",
        hw / "temp2_label": "junction", hw / "temp2_input": "88000",
        hw / "power1_average": "325000000", hw / "fan1_input": "1800",
    }
    for path, value in files.items():
        path.write_text(value + "\n")
    card = ROOT / "sys/class/drm/card1"
    card.mkdir(parents=True, exist_ok=True)
    if not (card / "device").exists():
        (card / "device").symlink_to(dev)
    (ROOT / "sys/class/drm/card1-DP-1").mkdir(exist_ok=True)
    dbg = ROOT / "sys/kernel/debug/dri/1"
    dbg.mkdir(parents=True, exist_ok=True)
    (dbg / "name").write_text("amdgpu dev=0000:03:00.0 unique=0000:03:00.0\n")
    (dbg / "amdgpu_gpu_recover").write_text("0\n")


def write_config():
    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config.OVERRIDE_TOML.write_text(f'[watchdog]\nllama_swap_url = "{BASE}"\nprobe_timeout = 2\n'
                                    'failures_before_action = 2\nstall_seconds = 60\n')


class ParseTests(unittest.TestCase):
    def test_prometheus_parse(self):
        m = prom.parse('# c\nfoo_total 3\nbar{b="2",a="1"} 4.5\nbad line\n')
        self.assertEqual(m[("foo_total", "")], 3.0)
        self.assertEqual(m[("bar", "a=1,b=2")], 4.5)

    def test_throughput_uses_busy_seconds(self):
        prev = {(metrics.GEN_TOKENS, ""): 100, (metrics.GEN_SECONDS, ""): 2,
                (metrics.PROMPT_TOKENS, ""): 0, (metrics.PROMPT_SECONDS, ""): 0}
        cur = {(metrics.GEN_TOKENS, ""): 400, (metrics.GEN_SECONDS, ""): 5,
               (metrics.PROMPT_TOKENS, ""): 0, (metrics.PROMPT_SECONDS, ""): 0}
        self.assertEqual(metrics.throughput(prev, cur), {"gen_tps": 100.0})


class DiagnosticIdTests(unittest.TestCase):
    def test_round_trip(self):
        for _ in range(200):
            diag_id = bundle.new_diagnostic_id()
            self.assertRegex(diag_id, r"^SBL-[0-9A-HJKMNP-TV-Z]{4}-[0-9A-HJKMNP-TV-Z]{4}$")
            self.assertTrue(bundle.valid_diagnostic_id(diag_id))
            self.assertTrue(bundle.valid_diagnostic_id(diag_id.lower()))

    def test_single_typo_is_caught(self):
        diag_id = bundle.new_diagnostic_id()
        raw = diag_id.replace("SBL-", "").replace("-", "")
        for i in range(7):
            for c in bundle.ALPHABET:
                if c == raw[i]:
                    continue
                typo = raw[:i] + c + raw[i + 1:]
                self.assertFalse(bundle.valid_diagnostic_id(typo), typo)


class RedactionTests(unittest.TestCase):
    def test_secrets_and_addresses(self):
        r = Redactor(b"salt", hostname="gpubox")
        text = ("key nd-" + "f" * 48 + " Authorization: Bearer abc.def.ghi hf_" + "x" * 30
                + " from 192.168.1.20 to 127.0.0.1 mac aa:bb:cc:dd:ee:ff on gpubox admin@example.com"
                + " password=hunter2")
        out = r.text(text)
        for leaked in ("nd-fff", "abc.def.ghi", "hf_xxx", "192.168.1.20", "aa:bb:cc", "gpubox",
                       "admin@example.com", "hunter2"):
            self.assertNotIn(leaked, out)
        self.assertIn("127.0.0.1", out)
        # The same address maps to the same token, so logs stay readable.
        self.assertEqual(r.text("192.168.1.20"), r.text("x 192.168.1.20").split()[-1])


class LadderTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config.WatchdogConfig()
        self.now = 100_000.0

    def test_restart_then_gpu_reset_then_reboot_then_degraded(self):
        s = watchdog.State()
        self.assertEqual(watchdog.decide(s, self.cfg, self.now, "api-unresponsive", True), "restart")
        s.restarts = [self.now - 10] * 3
        self.assertEqual(watchdog.decide(s, self.cfg, self.now, "api-unresponsive", True), "gpu-reset")
        s.gpu_resets = [self.now - 10]
        self.assertEqual(watchdog.decide(s, self.cfg, self.now, "api-unresponsive", True), "reboot")
        s.reboots = [self.now - 100, self.now - 50]
        self.assertEqual(watchdog.decide(s, self.cfg, self.now, "api-unresponsive", True), "degraded")

    def test_old_actions_fall_out_of_the_window(self):
        s = watchdog.State(restarts=[self.now - 1000] * 3)
        self.assertEqual(watchdog.decide(s, self.cfg, self.now, "api-unresponsive", True), "restart")

    def test_crash_loop_skips_restart_and_non_amd_skips_reset(self):
        s = watchdog.State()
        self.assertEqual(watchdog.decide(s, self.cfg, self.now, "crash-loop", False), "reboot")


class GpuTests(unittest.TestCase):
    def setUp(self):
        make_sysfs()

    def test_discover_skips_connectors(self):
        gpus = gpu.discover()
        self.assertEqual(len(gpus), 1)
        g = gpus[0]
        self.assertEqual((g.vendor, g.pci, g.vram_total), ("amd", "0000:03:00.0", 24 * 1024**3))

    def test_sample_amd(self):
        reading = gpu.sample_amd(gpu.discover()[0])
        self.assertEqual(reading["busy_percent"], 97)
        self.assertEqual(reading["temp_junction_c"], 88.0)
        self.assertEqual(reading["power_w"], 325.0)

    def test_debugfs_recover_path(self):
        path = gpu.debugfs_recover_path(gpu.discover()[0])
        self.assertEqual(path, ROOT / "sys/kernel/debug/dri/1/amdgpu_gpu_recover")


class SamplerTests(unittest.TestCase):
    def setUp(self):
        make_sysfs()
        write_config()
        config.METRICS_DB.unlink(missing_ok=True)
        FAKE.healthy = True
        FAKE.models = {"qwen": "ready"}
        FAKE.counters.update(prompt_tokens=0, prompt_seconds=0, gen_tokens=0, gen_seconds=0)
        FAKE.activity = []

    def test_samples_requests_and_restart(self):
        db = metrics.connect()
        self.addCleanup(db.close)
        s = metrics.Sampler(config.load_config(), db)
        s.sample_once()
        FAKE.counters.update(prompt_tokens=2000, prompt_seconds=1, gen_tokens=500, gen_seconds=10)
        FAKE.activity = [
            {"id": i, "timestamp": "2026-10-30T01:00:00Z", "model": "qwen", "resp_status_code": 200,
             "tokens": {"input_tokens": 100, "output_tokens": 50, "prompt_per_second": 900.0,
                        "tokens_per_second": 40.0 + i}, "duration_ms": 1500}
            for i in (1, 2, 3)]
        s.sample_once()
        gen = db.execute("SELECT value FROM samples WHERE metric='gen_tps'").fetchall()
        self.assertEqual(gen, [(50.0,)])
        gpu_busy = db.execute("SELECT labels, value FROM samples WHERE metric='gpu_busy_percent'").fetchall()
        self.assertIn(("gpu=0", 97.0), gpu_busy)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM requests").fetchone()[0], 3)

        s.sample_once()  # nothing new, no duplicates
        self.assertEqual(db.execute("SELECT COUNT(*) FROM requests").fetchone()[0], 3)

        FAKE.activity = [dict(FAKE.activity[0], id=1)]  # llama-swap restarted
        s.sample_once()
        s.sample_once()
        self.assertEqual(db.execute("SELECT COUNT(*) FROM requests").fetchone()[0], 4)

        summary = metrics.perf_summary(db, 0)
        self.assertEqual(summary[0]["model"], "qwen")
        self.assertEqual(summary[0]["requests"], 4)
        self.assertEqual(summary[0]["gen_tps_p50"], 41.5)  # 41, 41, 42, 43


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        make_sysfs()
        write_config()
        FAKE.healthy = True
        FAKE.models = {"qwen": "ready"}
        FAKE.counters.update(decode=10, processing=1)

    def test_stall_detection(self):
        wd = watchdog.Watchdog(config.load_config())
        self.assertIsNone(wd.check_models(1000.0))
        self.assertIsNone(wd.check_models(1030.0))
        problem = wd.check_models(1100.0)
        self.assertEqual(problem[0], "generation-stalled")
        FAKE.counters["decode"] = 11
        self.assertIsNone(wd.check_models(1130.0))

    def test_unresponsive_api_needs_consecutive_failures(self):
        wd = watchdog.Watchdog(config.load_config())
        FAKE.healthy = False
        self.assertIsNone(wd.check_swap())
        self.assertEqual(wd.check_swap()[0], "api-unresponsive")
        FAKE.healthy = True
        self.assertIsNone(wd.check_swap())
        self.assertEqual(wd.swap_failures, 0)

    def test_gpu_fault_pattern(self):
        self.assertTrue(watchdog.GPU_FAULT.search(
            "amdgpu 0000:03:00.0: amdgpu: ring gfx_0.0.0 timeout, signaled seq=1, emitted seq=3"))
        self.assertFalse(watchdog.GPU_FAULT.search("amdgpu: SMU driver if version matched"))


class BootCheckTests(unittest.TestCase):
    def setUp(self):
        write_config()
        config.SAFE_MODE_FLAG.unlink(missing_ok=True)
        config.WATCHDOG_STATE.unlink(missing_ok=True)
        config.RUNNING_MARKER.unlink(missing_ok=True)

    def test_unclean_boots_enter_safe_mode(self):
        cfg = config.load_config()
        self.assertEqual(watchdog.boot_check(cfg), "clean")
        results = [watchdog.boot_check(cfg) for _ in range(3)]  # marker never removed
        self.assertEqual(results, ["unclean", "unclean", "safe-mode"])
        self.assertTrue(config.SAFE_MODE_FLAG.exists())
        reasons = {json.loads((p / "incident.json").read_text())["reason"] for p in incident.list_incidents()}
        self.assertTrue({"unclean-shutdown", "boot-loop"} <= reasons)

    def test_clean_shutdown_clears_marker(self):
        cfg = config.load_config()
        watchdog.boot_check(cfg)
        watchdog.clean_shutdown()
        self.assertEqual(watchdog.boot_check(cfg), "clean")


class BundleTests(unittest.TestCase):
    def setUp(self):
        make_sysfs()
        write_config()
        FAKE.healthy = True
        FAKE.uploads.clear()

    def test_bundle_is_redacted_and_upload_needs_consent(self):
        inc = incident.capture("manual", {"note": "test"})
        self.assertTrue((inc / "llama-swap-logs.txt").exists())
        diag_id, path, manifest = bundle.build(inc, note="box at 10.1.2.3 hung")
        self.assertTrue(bundle.valid_diagnostic_id(diag_id))
        with tarfile.open(path) as tar:
            names = tar.getnames()
            self.assertIn(f"{diag_id}/manifest.json", names)
            logs = tar.extractfile(f"{diag_id}/incident/llama-swap-logs.txt").read().decode()
            meta = json.loads(tar.extractfile(f"{diag_id}/manifest.json").read())
        self.assertNotIn("nd-aaaa", logs)
        self.assertNotIn("10.1.2.3", meta["note"])
        self.assertEqual(meta["device_id"], config.device_id())

        config.save_consent("off", BASE)
        with self.assertRaises(bundle.UploadError):
            bundle.upload(path)
        self.assertEqual(FAKE.uploads, [])

        config.save_consent("manual")
        result = bundle.upload(path)
        self.assertEqual(result["diagnostic_id"], diag_id)
        headers, body = FAKE.uploads[0]
        self.assertEqual(headers["X-Sibyl-Diagnostic-Id"], diag_id)
        self.assertEqual(body, path.read_bytes())

    def test_plain_http_is_refused_off_loopback(self):
        with self.assertRaises(bundle.UploadError):
            bundle._check_endpoint("http://diag.example.com")
        self.assertEqual(bundle._check_endpoint("https://diag.example.com/"), "https://diag.example.com")

    def test_upload_pending_only_in_automatic_mode(self):
        config.save_consent("manual", BASE)
        _, path, _ = bundle.build(None)
        bundle.mark_pending(path)
        self.assertEqual(bundle.upload_pending(), [])
        config.save_consent("automatic")
        self.assertEqual(bundle.upload_pending(), [path.name])
        self.assertFalse(path.with_name(path.name + ".pending").exists())


if __name__ == "__main__":
    unittest.main()
