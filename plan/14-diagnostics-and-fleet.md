# Implementation Plan: 14-diagnostics-and-fleet.md — diagnostics, watchdog, opt-in upload, fleet service

This plan adds failure capture, performance metrics, a hardware inventory and an inference watchdog to the Sibyl host. Upload of diagnostics is opt-in. The upload path is the first part of a later fleet service for enterprise customers. The plan builds on `13-bootc-platform.md`.

## Goals

- Capture enough data at the moment of an inference failure to find the cause without access to the box.
- Record tokens per second, GPU load, temperature, power and memory per model, so that a change to the OS, the backend or a model shows up as a number.
- Bring inference back without an operator after a hang, a crash or a full kernel lockup.
- Send nothing off the box unless the owner turns uploads on.
- Give every bundle a diagnostic ID that the owner can quote to support.

## Components

All host parts are in the OS image. The code is Python standard library only, because the host has no pip and the watchdog must survive a broken upgrade.

| Part | Path | Runs as |
|---|---|---|
| `sibyl-diag` CLI and package | `os/rootfs/usr/bin/sibyl-diag`, `os/rootfs/usr/lib/sibyl/python/sibyl_diag/` | root |
| Metrics sampler | `sibyl-metrics.service` | root, sandboxed, writes only `/var/lib/sibyl/diagnostics` |
| Inference watchdog | `sibyl-watchdog.service` | root |
| Boot check | `sibyl-boot-check.service` | root, before `user@1001.service` |
| Upload queue | `sibyl-diag-upload.timer` | root, sandboxed |
| Hardware watchdog | `usr/lib/systemd/system.conf.d/60-sibyl-watchdog.conf` | systemd (PID 1) |
| Kernel panic settings | `usr/lib/sysctl.d/61-sibyl-panic.conf` | kernel |
| Persistent journal | `usr/lib/systemd/journald.conf.d/60-sibyl.conf` | journald |
| Defaults | `usr/share/sibyl/diagnostics.toml`, override in `/var/lib/sibyl/config/diagnostics.toml` | |

The watchdog and the sampler reach llama-swap on `127.0.0.1:18080`. `llama-swap.container` publishes that port on host loopback only. nftables accepts loopback traffic, and no rule forwards to the port, so Caddy stays the only remote entry point. Any local process can use this port without the API key. The host has no interactive users except the admin, so this plan accepts that risk.

## What the box records

The sampler writes to `/var/lib/sibyl/diagnostics/metrics.db` (SQLite) every 10 seconds. It keeps 7 days.

| Source | Data |
|---|---|
| amdgpu sysfs | Busy percent, VRAM and GTT use, edge, junction and memory temperature, power, power cap, fan, shader and memory clock |
| `nvidia-smi` | The same fields for NVIDIA cards |
| `/proc`, `statvfs` | Load, free memory, swap use, free space on `/var/lib/sibyl` |
| llama-swap `/running` | Loaded models and their state |
| llama-server `/metrics` through `/upstream/<model>/metrics` | Every `llamacpp:*` counter and gauge, plus generation and prompt tokens per second computed from counter deltas |
| llama-swap `/api/metrics/activity` | One row per request: model, status, input, cached and output tokens, prompt and generation tokens per second, duration, error |

The token speeds from the counters use busy seconds, not wall-clock time. They show the speed of the backend while it works, which is the number that a backend or quant change moves.

`sibyl-diag perf --since 24h` prints request count, errors, and p50 and p5 generation speed per model.

## Incidents

An incident is a directory under `/var/lib/sibyl/diagnostics/incidents/<UTC time>-<reason>/`. The watchdog captures it before it acts, because a restart or a reboot destroys most of the state. One failed capture step does not stop the others. The failures go to `capture-errors.txt`.

| File | Content |
|---|---|
| `incident.json` | Reason, details, boot ID, device ID, Sibyl version, GPU variant, inference image |
| `journal-kernel.txt` | Kernel log of the current boot, with amdgpu ring timeouts and resets |
| `journal-system.txt`, `journal-sibyl-user.txt`, `journal-llama-swap.txt` | Last 1 to 2 hours of system, user-unit and container logs |
| `journal-previous-boot*.txt`, `pstore/` | After a crash: the log of the boot that crashed and the kernel panic text |
| `llama-swap-logs.txt`, `llama-swap-state.json` | llama-swap and llama-server output, loaded models, last 200 requests, raw counters |
| `llama-swap.yaml` | The model commands: file, quant, context size, split mode, devices |
| `gpu-tools.txt` | `amd-smi`, `rocm-smi` or `nvidia-smi` output |
| `hardware.json` | The inventory below |
| `samples-1h.csv`, `requests-1h.csv`, `perf-24h.json` | Metrics around the failure |
| `units.txt`, `podman.txt`, `watchdog-state.json` | Unit status, container and image digests, recovery history |

The box keeps 50 incidents or 2 GB, whichever limit it reaches first. `sibyl-diag capture --note "..."` records one by hand.

## Hardware inventory

`sibyl-diag hardware` prints CPU model and relevant flags, memory, board and BIOS, each GPU with PCI ID, VBIOS, VRAM and PCIe link speed and width, disks, kernel and command line, bootc image digests for the booted, staged and rollback slots, watchdog devices, pstore backend, EFI and Secure Boot state. It leaves out serial numbers, MAC addresses and the machine ID.

## Watchdog

### Probes

The watchdog runs every 30 seconds. A probe must fail 3 times in a row before the watchdog acts.

| Probe | Problem name |
|---|---|
| `llama-swap.service` is `failed` | `service-failed` |
| `NRestarts` grew 3 times in 15 minutes | `crash-loop` |
| `GET /health` on llama-swap fails | `api-unresponsive` |
| `GET /upstream/<model>/health` or `/metrics` fails for a ready model | `model-unresponsive` |
| `requests_processing > 0` and no change in tokens, prompt tokens or decode batches for 300 seconds | `generation-stalled` |
| Kernel log line matches an amdgpu, NVIDIA Xid, i915 or soft-lockup pattern | `gpu-fault` (capture only) |

The watchdog does not probe while the unit is starting or stopping, because an image pull at start can take minutes. It also does nothing while the unit is `inactive`, because an operator stopped it.

### Recovery ladder

`watchdog.decide()` picks the next step from the recovery history:

1. Restart llama-swap. At most 3 times in 15 minutes. A crash loop skips this step, because systemd already restarts the unit.
2. Reset each AMD GPU through `/sys/kernel/debug/dri/<n>/amdgpu_gpu_recover`, then restart llama-swap. Once in 15 minutes. If the reset fails or hangs for 60 seconds, the driver is stuck, and the watchdog goes to step 3.
3. Reboot the host. At most 2 times in 6 hours.
4. Stop acting and report `degraded`. The box stays up, so that an operator can read the state.

`llama-swap.container` also sets `HealthOnFailure=kill`. Podman kills the container after 3 failed health checks, and `Restart=always` brings it back. This covers a dead llama-swap between watchdog probes.

### Full kernel lockup

A lockup stops the watchdog process too. These parts handle it, in order:

1. `kernel.softlockup_panic=1` and `kernel.panic_on_oops=1` turn a soft lockup or an oops into a panic.
2. The panic text goes to pstore in EFI variables. `kernel.panic=10` reboots 10 seconds later.
3. If the kernel cannot panic, the hardware watchdog resets the board. systemd pets `/dev/watchdog0` with `RuntimeWatchdogSec=60s`. `RebootWatchdogSec=5min` covers a reboot that hangs on a stuck driver.
4. At the next boot, `systemd-pstore` saves the panic text to `/var/lib/systemd/pstore`.
5. `sibyl-boot-check` finds the `running` marker that a clean shutdown removes. It records an `unclean-shutdown` incident with the journal of the crashed boot and the pstore files.
6. After more than 2 unclean boots in 6 hours, boot check writes `/var/lib/sibyl/config/safe-mode`. `llama-swap.container` has `ConditionPathExists=!` on that file, so inference does not start. The rest of the box comes up for diagnosis. `sibyl-diag safe-mode off` clears it.

The hardware watchdog needs a driver: `sp5100_tco` on AMD chipsets, `iTCO_wdt` on Intel. `sibyl-diag hardware` lists the watchdog devices. If the list is empty on a target board, step 3 does not work on that board.

## Consent and privacy

Local capture is always on and stays on the box. Upload has three modes. Only `sibyl-diag consent` writes the mode, to `/var/lib/sibyl/config/diagnostics-consent.json`, so an edit to the TOML file cannot turn uploads on.

| Mode | Behavior |
|---|---|
| `off` (default) | Nothing leaves the box. Bundles can still go to support on a USB stick. |
| `manual` | `sibyl-diag submit` uploads one bundle after it shows the size and the endpoint and asks for confirmation. |
| `automatic` | The watchdog bundles each incident and queues it. `sibyl-diag-upload.timer` sends the queue every hour. |

Every bundle is redacted before it is written. The redactor removes private keys, bearer tokens, Sibyl API keys, Hugging Face, OpenAI and GitHub tokens, `key=value` secrets and email addresses. It replaces IPv4, IPv6 and MAC addresses and the hostname with a keyed hash. The key is a random salt per device, so one address gives the same token in every file, and the token cannot be reversed. Container network addresses (`10.88.*`, `10.89.*`) and loopback stay as they are.

Bundles never contain prompts or responses. llama-swap request rows hold token counts and speeds only. Binary files that the redactor cannot read are left out.

The device ID is a random UUID in `/var/lib/sibyl/diagnostics/device-id`. It is not derived from the machine ID or any serial number.

The upload endpoint must use HTTPS. Plain HTTP is accepted for loopback only, for tests.

## Diagnostic ID

A diagnostic ID has the form `SBL-XXXX-XXXX`. It uses Crockford base32, which has no I, L, O or U, so it can be read over the phone. Seven characters are random (35 bits). The eighth is a check character with odd weights, so any single wrong character fails the check. The box makes the ID when it builds a bundle, so the ID exists with or without an upload. The ingest service treats the ID as a proposal and returns the ID that it stored.

## Ingest service, API v1

The service is not built yet. `diagnostics.toml` has an empty `endpoint`. The image build sets it when the service exists, and `sibyl-diag consent manual --endpoint` overrides it per box.

```
POST {endpoint}/v1/bundles
Content-Type: application/gzip
X-Sibyl-Diagnostic-Id: SBL-7K2M-9QXD
X-Sibyl-Device-Id: <uuid>
X-Sibyl-Version: <sibyl version>
X-Content-SHA256: <hex sha256 of body>

201 {"diagnostic_id": "SBL-7K2M-9QXD"}
```

The service must:

- Check the SHA-256 and the check character, and reject a body over 50 MB.
- Store the bundle encrypted at rest, keyed by diagnostic ID, with a retention limit (90 days is the starting proposal).
- Rate-limit per device ID and per source address, because the endpoint takes anonymous uploads.
- Parse `manifest.json`, `hardware.json` and `perf-24h.json` into searchable fields: GPU model, VBIOS, kernel, Sibyl version, inference image, model file, failure reason. This makes "every gfx1100 crash on image X" one query.
- Give support a lookup by diagnostic ID, and a way to delete all bundles of a device ID on request.

## Fleet service (long term)

The enterprise offering pairs Sibyl boxes with a cloud control plane. It manages configuration, updates and support for many boxes. The diagnostics work above is its first part: the device ID, the bundle format and the upload client carry over unchanged.

### Pairing and transport

- The TUI or `sibyl setup` shows a short pairing code. The admin enters it in the fleet console. The service issues a client certificate bound to the device ID.
- The box makes outbound connections only, over mTLS. No inbound port opens. A box behind NAT or a strict firewall still works.
- Unpairing deletes the certificate on both sides and sets the consent mode back to the owner's choice.
- Pairing implies consent for the data classes that the contract names. The box shows those classes during pairing.

### Agent

A `sibyl-agent` service on the host holds the connection. It reuses `sibyl_diag` for inventory, metrics and bundles. It accepts only commands from a fixed list, each with a typed payload:

| Command | Effect |
|---|---|
| `inventory`, `metrics` | Send the inventory and a metrics summary |
| `bundle` | Build and upload a bundle, with a diagnostic ID that support chose |
| `apply-config` | Write `hardware.yaml` or a model layout, then let the System API regenerate llama-swap configuration |
| `pull-model` | Download a model from an allowed source |
| `upgrade` | `bootc switch` to a pinned, signed image digest |
| `restart-service` | Restart one Sibyl unit |
| `safe-mode` | Turn safe mode on or off |

The agent has no remote shell. A support session that needs one is a separate, time-limited grant that the owner approves on the box.

### Fleet updates

- Rings: canary, early, broad. A ring gets a digest only after the previous ring reports healthy for a set time.
- Health is the same watchdog data: no `degraded` state, no unclean boots, and token speed within a tolerance of the ring's baseline for the same GPU and model.
- A box that fails health after an upgrade runs `bootc rollback` and reports. greenboot can enforce this rollback at boot. See the open items.
- Offline sites get the same digests as OCI archives, from `13-bootc-platform.md` phase 4.

### Console

- Box list with version, GPUs, loaded models, health and token speed.
- Incident list per box and across the fleet, grouped by failure reason and hardware.
- Configuration templates per group of boxes.
- Audit log of every command, with who sent it.

## Phases

| Phase | Scope | Exit check |
|---|---|---|
| D1 | This change: sampler, watchdog, boot check, incidents, bundles, consent, upload client, unit tests | `tests/diag` passes in CI. On the target host, `kill -STOP` on llama-server gives a `generation-stalled` incident and a restart. `echo c > /proc/sysrq-trigger` gives a reboot and an `unclean-shutdown` incident with pstore text. |
| D2 | System API routes for status, perf, incidents, bundle and consent. TUI page with the diagnostic ID and a consent switch. The setup wizard asks for the consent mode. | The wizard sets consent. The TUI shows the latest diagnostic ID. |
| D3 | Ingest service, API v1 above. Support lookup by diagnostic ID. | A box in `manual` mode uploads, and support finds the bundle by ID. |
| D4 | Pairing, `sibyl-agent`, inventory and metrics in the console. | Ten paired boxes show live health. |
| D5 | Remote configuration, model pulls, ring updates with rollback. | A bad image on the canary ring rolls back without a person. |

## Open items

- Phase D1 was tested on macOS against a fake sysfs and a fake llama-swap only. The GPU reset path, the hardware watchdog and pstore need a test on the 7900 XTX host before D1 closes.
- `kernel.panic_on_oops=1` reboots on any oops, including one that the box can survive. If this causes reboots on the target host, remove it and keep `softlockup_panic`.
- greenboot (`greenboot` package in Fedora) can roll back a bootc upgrade that fails a health check at boot. A health check script can call `sibyl-diag status`. This needs a test of greenboot with bootc on Fedora 44 first.
- kdump gives a full kernel dump but reserves memory at boot. pstore gives the panic text only. Decide after the first real crash if pstore is enough.
- An opt-in "capture failing request" mode can help model debugging, because some failures depend on the prompt. It needs llama-swap capture support (`/api/captures/{id}` exists in v266) and a separate consent level, because prompts can hold customer data.
- The watchdog restarts all of llama-swap. When a release of llama-swap can restart one model process, the first ladder step can unload only the failed model with `POST /api/models/unload/<model>`.
