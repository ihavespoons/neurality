# AGENTS.md

Orientation file for AI coding agents working in this repo. Keep it short and pointer-heavy — link to docs/code instead of duplicating specifics that drift.

## What this repo is

NeuralDrive is a bootable Debian-12 live distro (built with `live-build`) that turns an x86_64 host into a self-contained, OpenAI-compatible LLM inference server. Read first:

- `README.md` — product overview, features, hardware reqs, quick start
- `docs/user-guide/src/SUMMARY.md` — end-user documentation (mdbook)
- `docs/dev-guide/src/SUMMARY.md` — developer documentation (mdbook)
  - Architecture: `docs/dev-guide/src/architecture/`
  - Build system internals: `docs/dev-guide/src/build/`
- `plan/` — internal design docs (numbered 00–12). Source-of-truth for component-level design; **not** copied into the published mdbooks.
- `SIZE-REDUCTION-PLAN.md` — record of how the ISO got from ~32 GB to ~3.9 GB (open-webui `--no-deps`, NVIDIA headless packages, etc.). Read before touching package lists.

## Repo layout (one-liners)

- `config/` — live-build tree (hooks, includes.chroot, package-lists, archives, preseed)
- `config/hooks/live/` — chroot build hooks, numerically ordered (`01-setup-system` … `99-cleanup`)
- `config/includes.chroot/etc/systemd/system/` — `neuraldrive-*.service` units
- `config/includes.chroot/usr/lib/neuraldrive/` — runtime payload (TUI, System API, GPU detect, certs, monitors)
- `scripts/` — build/flash helpers (`docker-entrypoint.sh`, `neuraldrive-flash.sh`, `validate-config.sh`, `apply-branding.sh`, `download-models.sh`, …)
- `tests/` — boot/GPU/API tests (`test-boot.sh`, `test-gpu.sh`, `test_api.py`)
- `auto/` — `lb config`/`build`/`clean` wrappers consumed by live-build
- `Dockerfile`, `docker-compose.yml`, `build.sh`, `build_docker.sh` — build entry points (see below)
- `neuraldrive-build.yaml.example` — schema for optional custom-config builds

## Building the ISO

Building requires Debian + `live-build` + root + the ability to chroot/mknod. **On macOS or non-Debian Linux, use the Docker path.** Don't try to run `lb build` natively on macOS — it won't work.

### Docker build (the default; works on macOS and Linux)

```bash
./build_docker.sh                              # standard build
./build_docker.sh -f                           # force docker-image rebuild (no cache)
./build_docker.sh -s                           # set up workspace, drop into shell
./build_docker.sh -d                           # verbose; drop to shell on failure
./build_docker.sh -- neuraldrive-build.yaml    # custom config (see example file)
```

What that does, in order:

1. `docker compose build builder` — builds the `Dockerfile` (Debian bookworm + `live-build`, `debootstrap`, `squashfs-tools`, `xorriso`, `grub-pc-bin`, `grub-efi-amd64-bin`, `mtools`, `yq`).
2. `docker compose run --rm builder` with `platform: linux/amd64`, `cap_add: [SYS_ADMIN, MKNOD, SYS_CHROOT]`, `apparmor:unconfined`. Bind mounts `./:/src:ro` and `./output:/output`.
3. `scripts/docker-entrypoint.sh` runs inside the container:
   - **Patches `live-build`** in two places (grub-pc hybrid-MBR via `xorriso --grub2-mbr`, plus `part_msdos`+`search` modules in `binary_grub-pc`). Without these patches the ISO won't boot from USB. If you upgrade the base image and the patch greps stop matching, fix them here — don't work around it.
   - `cp -a /src/. /build/` so the heavy work happens in the container's writable overlay, not on the read-only bind mount.
   - Symlinks `/build/output → /output` so the finished ISO lands on the host without an extra copy.
   - Execs `./build.sh`.
4. `build.sh` runs `lb clean --all`, `lb config …`, `lb build`, then moves the ISO to `output/neuraldrive-<version>.iso` and writes `SHA256SUMS`.

**Output**: `./output/neuraldrive-<version>.iso` (+ `SHA256SUMS`) on the host filesystem.

**Version derivation** (`build.sh` top): `$NEURALDRIVE_VERSION` env > exact `vX.Y.Z` git tag on HEAD > `dev-YYYY.MM.DD-<short-sha>`. To cut a clean release ISO, tag first (e.g. `./scripts/tag-release.sh`) — otherwise expect a `dev-…` filename.

**Build time**: 30–90 min on native Debian/amd64 per README. On Apple Silicon under Rosetta/QEMU emulation, expect roughly 1.5–3× longer and very network-dependent (apt + pip + `ollama pull` all hit the wire).

**Disk**: the chroot, squashfs, and intermediate artifacts (~10–20 GB peak) live in Docker Desktop's VM disk, not in `./`. Only the final ISO (~3.9 GB target) is written to `./output/`. Make sure Docker Desktop has enough VM disk allocated.

**Privileges**: `./build_docker.sh` does **not** need `sudo` — Docker grants the required capabilities via `cap_add`. The native `./build.sh` path *does* need `sudo` (see `README.md` "Building from source").

### Native build (Debian 12 host only)

```bash
sudo apt install live-build debootstrap squashfs-tools xorriso grub-pc-bin grub-efi-amd64-bin
sudo ./build.sh
# or: sudo ./build.sh neuraldrive-build.yaml
```

This path **skips the Docker entrypoint's live-build patches**. If you hit "GRUB loading…" hangs from USB, port those patches into the host's `/usr/lib/live/build/binary_iso` and `binary_grub-pc` (see `scripts/docker-entrypoint.sh`).

### Common build pitfalls

- **GRUB hangs at boot** → live-build patches missing (see entrypoint).
- **ISO is huge (~30 GB)** → something pulled torch/CUDA transitively. Read `SIZE-REDUCTION-PLAN.md` before changing package lists or pip installs in the chroot hooks.
- **`apt install` failures mid-build** → usually upstream archive flakiness; `./build_docker.sh -d` drops to a shell on failure so you can inspect `/build/chroot/` and `/build/.build/`.
- **macOS bind mount weirdness** → entrypoint already copies `/src` into the container, so building from an external drive (`/Volumes/…`) is fine; the chroot never touches the host filesystem.

### Flashing the result

`scripts/neuraldrive-flash.sh <iso> /dev/sdX` (Linux), or `dd` on macOS, or Balena Etcher. See `README.md` "Quick start" for the canonical commands.

## Testing

- `tests/test-boot.sh` — boot-flow test
- `tests/test-gpu.sh` — GPU detection test
- `tests/test_api.py` — API surface test
- `tests/diag/` — unit tests for the Sibyl `sibyl-diag` diagnostics and watchdog package (`python3 -m unittest discover -s tests/diag`). CI runs them in `.github/workflows/bootc.yml` before the OS image builds. Design: `plan/14-diagnostics-and-fleet.md`.

There is no CI-enforced lint/format gate in this repo today. If you add one, document it here.

## Conventions worth knowing

- **Don't duplicate drift-prone specifics in this file.** Service names, port numbers, package lists, cap values, version strings, chapter counts — link to the code or docs instead.
- **Numbered prefixes are load order**, not arbitrary: `config/hooks/live/NN-*.chroot` and `plan/NN-*.md` are read/executed in order. Pick a free number when adding.
- Runtime code under `config/includes.chroot/usr/lib/neuraldrive/` is Python (Textual TUI, FastAPI System API). The `webui/` directory is intentionally empty — Open WebUI is installed via pip in a chroot hook, not vendored.
- Persistent design discussion lives in `plan/`. Short-lived implementation plans live in `.sisyphus/plans/` (gitignored).

## When in doubt

Read `docs/dev-guide/src/build/` (especially `live-build.md`, `docker.md`, `hooks.md`) and the matching `plan/` chapter before changing build, hook, or service behavior.
