"""Inference watchdog.

The watchdog probes llama-swap and each loaded llama-server every
probe_interval seconds. When a probe fails failures_before_action times in a
row, it captures an incident first and then climbs a recovery ladder:

1. Restart llama-swap, at most max_restarts times per restart_window.
2. Reset the GPU through amdgpu debugfs, once per restart_window, then restart.
3. Reboot the host, at most max_reboots times per reboot_window.
4. Stop acting and report "degraded", so a broken box stays up for diagnosis
   instead of rebooting forever.

A full kernel lockup stops this process too. The hardware watchdog that
systemd arms (RuntimeWatchdogSec in system.conf.d/60-sibyl-watchdog.conf)
resets the board in that case, and `boot-check` records the crash at the
next boot.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field

from . import config, gpu, host, incident, prom
from .llamaswap import LlamaSwap, LlamaSwapError
from .sdnotify import notify

# Kernel messages that mean the GPU or its driver failed.
GPU_FAULT = re.compile(
    r"amdgpu.*(ring \S+ timeout|GPU reset|VRAM is lost|page fault|GPU hang|"
    r"failed to initialize parser|IB test failed|SMU is not ready|Fatal error)"
    r"|NVRM: Xid|i915.*GPU HANG|xe .*(reset|hang)|BUG: soft lockup|hung_task|"
    r"Out of memory: Killed process",
    re.I)

# Counters that move while llama-server does any work on a request.
# n_decode_total moves once per batch, so it also covers a long prompt.
PROGRESS_COUNTERS = ("llamacpp:tokens_predicted_total", "llamacpp:prompt_tokens_total",
                     "llamacpp:n_decode_total")

# Unit states where llama-swap is starting or stopping. An image pull at start
# can take minutes, so the HTTP probes wait.
TRANSITIONS = ("activating", "deactivating", "reloading", "refreshing")


@dataclass
class State:
    restarts: list[float] = field(default_factory=list)
    gpu_resets: list[float] = field(default_factory=list)
    reboots: list[float] = field(default_factory=list)
    unclean_boots: list[float] = field(default_factory=list)
    crashes: list[float] = field(default_factory=list)
    last_incident: dict[str, float] = field(default_factory=dict)
    degraded: bool = False
    journal_cursor: str = ""
    nrestarts: int = -1
    last_action: str = ""
    last_problem: str = ""

    @classmethod
    def load(cls) -> State:
        try:
            data = json.loads(config.WATCHDOG_STATE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self) -> None:
        config.write_private(config.WATCHDOG_STATE, json.dumps(self.__dict__, indent=2) + "\n")

    def prune(self, now: float, keep: float) -> None:
        for name in ("restarts", "gpu_resets", "reboots", "unclean_boots", "crashes"):
            setattr(self, name, [t for t in getattr(self, name) if now - t < keep])


def _recent(times: list[float], now: float, window: int) -> int:
    return sum(1 for t in times if now - t < window)


def decide(state: State, cfg: config.WatchdogConfig, now: float, problem: str, has_amd: bool) -> str:
    """Pick the next rung of the recovery ladder. Pure, so the tests cover it."""
    restarts = _recent(state.restarts, now, cfg.restart_window)
    # systemd already restarts a crash loop, so another restart does not help.
    if problem != "crash-loop" and restarts < cfg.max_restarts:
        return "restart"
    if cfg.gpu_reset and has_amd and _recent(state.gpu_resets, now, cfg.restart_window) == 0:
        return "gpu-reset"
    if cfg.reboot and _recent(state.reboots, now, cfg.reboot_window) < cfg.max_reboots:
        return "reboot"
    return "degraded"


@dataclass
class ModelProbe:
    failures: int = 0
    progress: float = -1.0
    stalled_since: float = 0.0


class Watchdog:
    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.wd = cfg.watchdog
        self.swap = LlamaSwap(self.wd.llama_swap_url, timeout=self.wd.probe_timeout)
        self.state = State.load()
        self.gpus = gpu.discover()
        self.swap_failures = 0
        self.models: dict[str, ModelProbe] = {}

    # Probes. Each returns (problem, details) or None.

    def check_unit(self, active: str) -> tuple[str, dict] | None:
        nrestarts = host.unit_property("llama-swap.service", "NRestarts")
        if nrestarts.isdigit():
            n = int(nrestarts)
            if 0 <= self.state.nrestarts < n:
                self.state.crashes += [time.time()] * (n - self.state.nrestarts)
            self.state.nrestarts = n
        if _recent(self.state.crashes, time.time(), self.wd.restart_window) >= self.wd.max_restarts:
            return "crash-loop", {"crashes_in_window": len(self.state.crashes), "active": active}
        if active == "failed":
            return "service-failed", {"active": active}
        return None

    def check_swap(self) -> tuple[str, dict] | None:
        try:
            ok = self.swap.health()
        except LlamaSwapError as exc:
            ok, err = False, str(exc)
        else:
            err = "" if ok else "unexpected /health body"
        if ok:
            self.swap_failures = 0
            return None
        self.swap_failures += 1
        if self.swap_failures >= self.wd.failures_before_action:
            return "api-unresponsive", {"error": err, "failures": self.swap_failures}
        return None

    def check_models(self, now: float) -> tuple[str, dict] | None:
        try:
            running = self.swap.running()
        except LlamaSwapError:
            return None  # check_swap reports this case.
        ready = {str(e.get("model")) for e in running if e.get("state") == "ready"}
        self.models = {m: p for m, p in self.models.items() if m in ready}
        for model in sorted(ready):
            probe = self.models.setdefault(model, ModelProbe())
            try:
                self.swap.upstream_health(model)
                metrics = self.swap.upstream_metrics(model)
            except LlamaSwapError as exc:
                probe.failures += 1
                if probe.failures >= self.wd.failures_before_action:
                    return "model-unresponsive", {"model": model, "error": str(exc)}
                continue
            probe.failures = 0
            busy = prom.value(metrics, "llamacpp:requests_processing") > 0
            progress = sum(prom.value(metrics, c) for c in PROGRESS_COUNTERS)
            if busy and progress == probe.progress:
                probe.stalled_since = probe.stalled_since or now
                if now - probe.stalled_since >= self.wd.stall_seconds:
                    return "generation-stalled", {"model": model,
                                                  "stalled_seconds": int(now - probe.stalled_since)}
            else:
                probe.stalled_since = 0.0
            probe.progress = progress
        return None

    def check_kernel(self) -> list[str]:
        args = ["journalctl", "-k", "--no-pager", "-o", "cat", "--show-cursor"]
        args += ["--after-cursor", self.state.journal_cursor] if self.state.journal_cursor else ["-n", "0"]
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=30).stdout
        except (subprocess.SubprocessError, OSError):
            return []
        hits = []
        for line in out.splitlines():
            if line.startswith("-- cursor: "):
                self.state.journal_cursor = line.removeprefix("-- cursor: ").strip()
            elif GPU_FAULT.search(line):
                hits.append(line)
        return hits

    # Actions.

    def capture(self, reason: str, details: dict) -> None:
        now = time.time()
        if now - self.state.last_incident.get(reason, 0) < self.wd.incident_cooldown:
            return
        self.state.last_incident[reason] = now
        self.state.save()
        try:
            path = incident.capture(reason, details)
        except Exception as exc:  # noqa: BLE001 - the watchdog must keep running
            print(f"sibyl-watchdog: incident capture failed: {exc}", flush=True)
            return
        print(f"sibyl-watchdog: incident {path.name}", flush=True)
        from . import bundle
        if config.load_consent().mode == "automatic":
            _, bpath, _ = bundle.build(path)
            bundle.mark_pending(bpath)

    def gpu_reset(self) -> bool:
        ok = False
        for g in self.gpus:
            path = gpu.debugfs_recover_path(g) if g.vendor == "amd" else None
            if not path:
                continue
            try:
                subprocess.run(["cat", str(path)], capture_output=True, timeout=60, check=True)
                ok = True
            except (subprocess.SubprocessError, OSError) as exc:
                print(f"sibyl-watchdog: GPU reset of {g.pci} failed: {exc}", flush=True)
                return False
        return ok

    def act(self, problem: str, details: dict) -> None:
        now = time.time()
        has_amd = any(g.vendor == "amd" for g in self.gpus)
        action = decide(self.state, self.wd, now, problem, has_amd)
        self.state.last_problem, self.state.last_action = problem, action
        print(f"sibyl-watchdog: {problem} {json.dumps(details)} -> {action}", flush=True)
        self.capture(problem, {**details, "action": action})

        if action == "gpu-reset":
            self.state.gpu_resets.append(now)
            if not self.gpu_reset():
                # A reset that hangs or fails means the driver is wedged.
                action = "reboot" if self.wd.reboot else "degraded"
                self.state.last_action = action
        if action in ("restart", "gpu-reset"):
            self.state.restarts.append(now)
            host.user_systemctl("restart", "--no-block", "llama-swap.service", timeout=30)
        elif action == "reboot":
            self.state.reboots.append(now)
            self.state.save()
            subprocess.run(["systemctl", "reboot"], timeout=60)
        elif action == "degraded":
            self.state.degraded = True
        self.swap_failures = 0
        self.models.clear()
        self.state.save()

    def tick(self) -> None:
        now = time.time()
        self.state.prune(now, max(self.wd.reboot_window, self.wd.restart_window))
        faults = self.check_kernel()
        if faults:
            self.capture("gpu-fault", {"kernel": faults[-20:]})
        if config.SAFE_MODE_FLAG.exists():
            self.state.save()
            return
        active = host.unit_property("llama-swap.service", "ActiveState")
        problem = self.check_unit(active)
        if not problem and active in TRANSITIONS:
            self.swap_failures = 0
            self.state.save()
            return
        if not problem and active == "inactive":
            # Stopped on purpose, for example from Arcane. Do not fight the operator.
            self.state.save()
            return
        problem = problem or self.check_swap() or self.check_models(now)
        if problem:
            if not self.state.degraded:
                self.act(*problem)
        elif self.state.degraded:
            print("sibyl-watchdog: inference is healthy again", flush=True)
            self.state.degraded = False
        self.state.save()


def run_daemon(cfg: config.Config) -> None:
    wd = Watchdog(cfg)
    notify("READY=1")
    while True:
        started = time.monotonic()
        if cfg.watchdog.enabled:
            try:
                wd.tick()
            except Exception as exc:  # noqa: BLE001 - log and keep probing
                print(f"sibyl-watchdog: tick failed: {type(exc).__name__}: {exc}", flush=True)
        status = "degraded" if wd.state.degraded else "safe mode" if config.SAFE_MODE_FLAG.exists() else "ok"
        notify(f"WATCHDOG=1\nSTATUS={status}")
        time.sleep(max(1.0, cfg.watchdog.probe_interval - (time.monotonic() - started)))


def boot_check(cfg: config.Config) -> str:
    """Run at boot. Record an unclean previous shutdown, and stop a boot loop."""
    state = State.load()
    now = time.time()
    state.prune(now, cfg.watchdog.reboot_window)
    result = "clean"
    if config.RUNNING_MARKER.exists():
        result = "unclean"
        state.unclean_boots.append(now)
        details = {"watchdog_reboot": bool(state.reboots and now - state.reboots[-1] < 900)}
        state.save()
        incident.capture("unclean-shutdown", details, previous_boot=True)
    # Restart counters belong to the previous boot.
    state.nrestarts = -1
    state.crashes = []
    state.degraded = False
    state.journal_cursor = ""
    crashes = _recent(state.unclean_boots, now, cfg.watchdog.reboot_window)
    if crashes > cfg.watchdog.max_reboots and not config.SAFE_MODE_FLAG.exists():
        config.SAFE_MODE_FLAG.write_text(f"boot loop: {crashes} unclean boots\n")
        incident.capture("boot-loop", {"unclean_boots": crashes})
        result = "safe-mode"
    state.save()
    config.RUNNING_MARKER.parent.mkdir(parents=True, exist_ok=True)
    config.RUNNING_MARKER.write_text(host.boot_id() + "\n")
    return result


def clean_shutdown() -> None:
    config.RUNNING_MARKER.unlink(missing_ok=True)
