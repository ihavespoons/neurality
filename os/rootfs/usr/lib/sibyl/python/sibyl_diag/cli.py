"""sibyl-diag command line."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from contextlib import closing
from datetime import datetime

from . import bundle, config, hardware, incident, metrics, watchdog


def _since(raw: str) -> int:
    m = re.fullmatch(r"(\d+)([mhd])", raw)
    if not m:
        raise argparse.ArgumentTypeError("use a duration such as 30m, 24h or 7d")
    return int(time.time()) - int(m.group(1)) * {"m": 60, "h": 3600, "d": 86400}[m.group(2)]


def _print(data: object, as_json: bool) -> None:
    if as_json or not isinstance(data, list):
        print(json.dumps(data, indent=2, default=str))
        return
    for row in data:
        print("  ".join(f"{k}={v}" for k, v in row.items()))


def cmd_status(args) -> int:
    consent = config.load_consent()
    state = watchdog.State.load()
    incidents = incident.list_incidents()
    out = {
        "version": config.sibyl_version(),
        "device_id": config.device_id(),
        "upload_consent": consent.mode,
        "upload_endpoint": bundle.endpoint() or None,
        "safe_mode": config.SAFE_MODE_FLAG.read_text().strip() if config.SAFE_MODE_FLAG.exists() else False,
        "watchdog": {
            "degraded": state.degraded,
            "last_problem": state.last_problem,
            "last_action": state.last_action,
            "restarts_recent": len(state.restarts),
            "reboots_recent": len(state.reboots),
            "unclean_boots_recent": len(state.unclean_boots),
        },
        "incidents": len(incidents),
        "latest_incident": incidents[-1].name if incidents else None,
    }
    try:
        with closing(metrics.connect()) as db:
            out["perf_1h"] = metrics.perf_summary(db, int(time.time()) - 3600)
    except Exception as exc:  # noqa: BLE001 - status must always print
        out["perf_1h"] = f"<unavailable: {exc}>"
    print(json.dumps(out, indent=2))
    return 0


def cmd_hardware(args) -> int:
    print(json.dumps(hardware.inventory(), indent=2, default=str))
    return 0


def cmd_perf(args) -> int:
    with closing(metrics.connect()) as db:
        _print(metrics.perf_summary(db, args.since, args.model), args.json)
    return 0


def cmd_incidents(args) -> int:
    for path in incident.list_incidents():
        try:
            meta = json.loads((path / "incident.json").read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        print(f"{path.name}  {meta.get('reason', '?')}  {json.dumps(meta.get('details', {}))}")
    return 0


def cmd_capture(args) -> int:
    path = incident.capture(args.reason, {"note": args.note} if args.note else {})
    print(path)
    return 0


def _resolve_incident(name: str | None, latest: bool):
    incidents = incident.list_incidents()
    if latest:
        if not incidents:
            sys.exit("No incidents recorded.")
        return incidents[-1]
    if name:
        for path in incidents:
            if path.name == name or path.name.startswith(name):
                return path
        sys.exit(f"No incident matches {name}.")
    return None


def cmd_bundle(args) -> int:
    inc = _resolve_incident(args.incident, args.latest)
    if inc is None:
        # No incident named: take a fresh snapshot so the bundle has logs.
        inc = incident.capture("manual", {"note": args.note} if args.note else {})
    diag_id, path, manifest = bundle.build(inc, note=args.note or "")
    print(f"Diagnostic ID: {diag_id}")
    print(f"Bundle: {path} ({path.stat().st_size} bytes, redacted)")
    if args.list:
        for entry in manifest:
            print(f"  {entry['bytes']:>10}  {entry['path']}")
    return 0


def cmd_submit(args) -> int:
    if args.bundle:
        path = config.BUNDLE_DIR / (args.bundle if args.bundle.endswith(".tar.gz") else f"{args.bundle}.tar.gz")
    else:
        bundles = sorted(config.BUNDLE_DIR.glob("SBL-*.tar.gz"), key=lambda p: p.stat().st_mtime)
        if not bundles:
            sys.exit("No bundles. Create one with `sibyl-diag bundle`.")
        path = bundles[-1]
    if not path.exists():
        sys.exit(f"No such bundle: {path}")
    if not args.yes:
        print(f"This sends {path.name} ({path.stat().st_size} bytes) to {bundle.endpoint() or '<no endpoint>'}.")
        print("The bundle is redacted. List its files with `tar tzf " + str(path) + "`.")
        if input("Send it? [y/N] ").strip().lower() != "y":
            return 1
    try:
        result = bundle.upload(path)
    except bundle.UploadError as exc:
        sys.exit(str(exc))
    print(f"Sent. Diagnostic ID: {result.get('diagnostic_id', path.name.removesuffix('.tar.gz'))}")
    return 0


def cmd_consent(args) -> int:
    if args.mode is None:
        c = config.load_consent()
        changed = datetime.fromtimestamp(c.changed_at).isoformat() if c.changed_at else "never"
        print(f"mode={c.mode} endpoint={bundle.endpoint() or '<none>'} changed={changed}")
        return 0
    if args.endpoint:
        bundle._check_endpoint(args.endpoint)
    c = config.save_consent(args.mode, args.endpoint)
    print(f"Upload consent is now {c.mode}.")
    return 0


def cmd_safe_mode(args) -> int:
    if args.action == "on":
        config.SAFE_MODE_FLAG.parent.mkdir(parents=True, exist_ok=True)
        config.SAFE_MODE_FLAG.write_text("set by operator\n")
        print("Safe mode is on. llama-swap does not start at the next boot, and the watchdog does not act.")
    elif args.action == "off":
        config.SAFE_MODE_FLAG.unlink(missing_ok=True)
        state = watchdog.State.load()
        state.unclean_boots, state.reboots, state.degraded = [], [], False
        state.save()
        print("Safe mode is off. Start inference with: systemctl --user -M sibyl@ start llama-swap.service")
    else:
        print(config.SAFE_MODE_FLAG.read_text().strip() if config.SAFE_MODE_FLAG.exists() else "off")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sibyl-diag", description="Sibyl diagnostics and watchdog")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="watchdog state, consent and recent speed").set_defaults(fn=cmd_status)
    sub.add_parser("hardware", help="hardware and software inventory").set_defaults(fn=cmd_hardware)

    s = sub.add_parser("perf", help="tokens per second per model")
    s.add_argument("--since", type=_since, default=_since("24h"))
    s.add_argument("--model")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_perf)

    sub.add_parser("incidents", help="list recorded incidents").set_defaults(fn=cmd_incidents)

    s = sub.add_parser("capture", help="record an incident now")
    s.add_argument("--reason", default="manual")
    s.add_argument("--note")
    s.set_defaults(fn=cmd_capture)

    s = sub.add_parser("bundle", help="create a redacted bundle with a diagnostic ID")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--incident", help="incident directory name or prefix")
    g.add_argument("--latest", action="store_true", help="use the latest incident")
    s.add_argument("--note", help="what you saw, in your own words")
    s.add_argument("--list", action="store_true", help="list the files in the bundle")
    s.set_defaults(fn=cmd_bundle)

    s = sub.add_parser("submit", help="upload a bundle (needs consent)")
    s.add_argument("bundle", nargs="?", help="diagnostic ID; default is the newest bundle")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_submit)

    s = sub.add_parser("consent", help="show or set upload consent")
    s.add_argument("mode", nargs="?", choices=config.UPLOAD_MODES)
    s.add_argument("--endpoint")
    s.set_defaults(fn=cmd_consent)

    s = sub.add_parser("safe-mode", help="stop inference from starting at boot")
    s.add_argument("action", nargs="?", choices=("on", "off", "status"), default="status")
    s.set_defaults(fn=cmd_safe_mode)

    # Entry points for the systemd units.
    for name, fn in (
        ("metrics-daemon", lambda a: metrics.run_daemon(config.load_config())),
        ("watchdog-daemon", lambda a: watchdog.run_daemon(config.load_config())),
        ("boot-check", lambda a: print(watchdog.boot_check(config.load_config())) or 0),
        ("clean-shutdown", lambda a: watchdog.clean_shutdown() or 0),
        ("upload-pending", lambda a: print("\n".join(bundle.upload_pending())) or 0),
    ):
        sub.add_parser(name, help=argparse.SUPPRESS).set_defaults(fn=fn)

    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
