# Implementation Plan: 13-bootc-platform.md — Fedora bootc host, containerized ROCm inference

This plan replaces the Debian live-build host and the Ollama runtime. It records the decisions from the October 2026 design rounds. It supersedes `01-base-system.md`, `03-llm-runtime.md` and the host parts of `04-storage-persistence.md` and `05-security.md` once the new build reaches parity.

## Decisions

| Area | Decision | Reason |
|---|---|---|
| Host OS | Fedora bootc (`quay.io/fedora/fedora-bootc:44`) | The OS is a Containerfile. Updates are atomic, with rollback. The kernel is current for gfx1100. |
| Container runtime | Podman with Quadlet, rootless | Fedora default. Quadlet units start at boot without a daemon. |
| Container UI | Arcane (BSD-3-Clause) | Compose editor, Git Sync, native Podman and Quadlet pages. Dockhand is source-available (BSL 1.1), so it is out. |
| Inference engine | llama.cpp `llama-server` behind llama-swap | Best single-stream speed for GGUF on RDNA3. llama-swap swaps models and places them on GPUs. |
| GPU backends | HIP (ROCm) and Vulkan images, both shipped | Results on RDNA3 differ by quant and context size. A benchmark picks one per model. |
| Model format | GGUF only | Fits 24 GB per card. No vLLM. |
| Multi-GPU | Per-model layout through the llama-swap matrix router | One model can span both cards, or two models can run on one card each. |
| Other GPUs | Separate image variants: `-amd`, `-nvidia`, `-intel` | Only the NVIDIA variant carries signed out-of-tree kernel modules. |
| Base for service images | Wolfi where possible | Smaller images with fewer CVEs. ROCm images stay on the AMD base, because Wolfi does not package ROCm. |
| Boot | Live USB with persistence, and install to disk | Try it from USB, then install. |
| Disks | Guided installer, LUKS with TPM2 unlock, model storage pools, USB persistence | See "Storage". |
| Hardening | SELinux enforcing, Secure Boot, signed images, CIS-style lockdown | See "Security". |
| Updates | GHCR registry, plus offline bundles | `bootc upgrade` online. `bootc switch` to an OCI archive offline. |
| CI | GitHub Actions, SHA-pinned actions, cosign keyless signing | Images and ISOs build in CI. |
| Model download | Hugging Face GGUF picker in the System API and TUI, plus an Ollama-compatible API shim | Existing Ollama clients keep working. |

## Repository layout

The new build lives beside the live-build tree until it reaches parity. Then `config/`, `auto/`, `build.sh` and the Debian `Dockerfile` are removed.

```
os/
  Containerfile              # bootc host image, ARG GPU_VARIANT=amd|nvidia|intel
  packages/                  # dnf package lists per variant
  rootfs/                    # files copied into / of the host image
containers/
  llama-swap/                # Containerfile.hip, Containerfile.vulkan
  caddy/                     # Wolfi-based TLS front door
  system-api/                # Wolfi-based FastAPI management API
  tui/                       # Wolfi-based Textual TUI
scripts/bootc/               # local build helpers, ISO creation, benchmark
.github/workflows/bootc.yml  # build, sign, push
```

## Runtime layout

All NeuralDrive containers run rootless under one system user, `neuraldrive` (UID 1001). Rootless Podman confines each container to that user's subordinate UID range. A container escape gives an unprivileged user, not root.

Arcane uses the same user's Podman socket. It therefore sees and controls the core containers. It runs with `SecurityLabelDisable=true`, because the Arcane docs require it on SELinux hosts. The socket proxy from the earlier design is not used. A proxy cannot filter the Quadlet and D-Bus calls that Arcane needs, and rootless Podman already removes root from the blast radius.

systemd owns the core services. The Quadlet files are in the OS image under `/etc/containers/systemd/users/1001/`. Arcane can stop a core container, but the next boot or `systemctl --user restart` brings it back from the image definition. User stacks that Arcane creates live under `/var/lib/neuraldrive/stacks`.

| Container | Image | Port on the `neuraldrive` network | Exposed on the host |
|---|---|---|---|
| `llama-swap` | `ghcr.io/ihavespoons/neuraldrive-llama-swap:<ver>-hip` or `-vulkan` | 8080 | No |
| `open-webui` | `ghcr.io/open-webui/open-webui` (pinned digest) | 8080 | No |
| `system-api` | `ghcr.io/ihavespoons/neuraldrive-system-api` | 3001 | No |
| `arcane` | `ghcr.io/getarcaneapp/manager` (pinned digest) | 3552 | No |
| `caddy` | `ghcr.io/ihavespoons/neuraldrive-caddy` | 10443, 18443, 19443 | Yes, through nftables redirect |

nftables redirects host port 443 to 10443 (Open WebUI), 8443 to 18443 (OpenAI API and System API), and 9443 to 19443 (Arcane). The input chain accepts the high ports only for redirected connections. This keeps `net.ipv4.ip_unprivileged_port_start` at its default of 1024.

## Inference

### Images

`containers/llama-swap/Containerfile.hip` builds llama.cpp for `gfx1100` only, on `rocm/dev-ubuntu-24.04:<ROCm>-complete`. A single-target build compiles faster and the kernels match the card. The runtime stage keeps only the HIP runtime libraries that `llama-server` links against.

`containers/llama-swap/Containerfile.vulkan` builds llama.cpp with `GGML_VULKAN=ON` on Ubuntu 26.04. Mesa RADV is the Vulkan driver. The image has no ROCm userspace. It uses Ubuntu, not Wolfi, to match the upstream llama.cpp Vulkan build.

Both images add the llama-swap release binary and run as UID 10001 inside the container.

Build arguments:

| Argument | Default | Purpose |
|---|---|---|
| `LLAMA_CPP_REF` | pinned `bNNNNN` tag | llama.cpp release |
| `LLAMA_SWAP_VERSION` | pinned release | llama-swap release |
| `ROCM_VERSION` | `7.2.4` | ROCm base. Upstream llama.cpp tests 7.2.x. ROCm 7.14 is a benchmark candidate. |
| `GPU_TARGETS` | `gfx1100` | RDNA3 Navi 31 |
| `GGML_HIP_RCCL` | `OFF` | RCCL helps only `--split-mode tensor`. Upstream disables it by default. |

### Multi-GPU

llama-swap uses the `matrix` router. The matrix lists the model combinations that can run at the same time. Models pinned to different cards form one set, and a model that spans both cards forms a set on its own. When a request arrives, llama-swap evicts the models outside the cheapest set that contains the requested model. The System API generates the sets from the layout of each model.

The `capacity` router, which tracks VRAM per card, is on llama-swap `main` but not in release v266. A v266 binary rejects it with `unknown router "capacity" (valid: group, matrix)`. When a release includes it, the System API can switch to it.

| Layout | llama-server flags | Pools used |
|---|---|---|
| Span both cards | `--split-mode layer --device ROCm0,ROCm1` | `gpu0` and `gpu1` |
| Pin to card 0 | `--split-mode none --device ROCm0` | `gpu0` |
| Pin to card 1 | `--split-mode none --device ROCm1` | `gpu1` |
| Tensor parallel (experimental) | `--split-mode tensor -fa on -ctk f16 -ctv f16` | `gpu0` and `gpu1` |

`--split-mode tensor` is experimental upstream, and upstream gives no performance guarantee outside CUDA (`docs/multi-gpu.md` in llama.cpp). The benchmark decides if it beats `layer` on two 7900 XTX cards.

Device names differ per backend: `ROCm0` for HIP and `Vulkan0` for Vulkan. The System API writes the llama-swap configuration, so it fills in the right names.

### Benchmark

`scripts/bootc/bench.sh` runs `llama-bench` from both images against each installed model. It records prompt-processing and token-generation speed at 512, 8192 and 32768 tokens of context. The System API stores the faster backend per model and writes it into the llama-swap configuration.

### Embeddings for Open WebUI

Open WebUI uses `RAG_EMBEDDING_ENGINE=ollama` today (`config/hooks/live/05-generate-configs.chroot`). The new configuration uses `RAG_EMBEDDING_ENGINE=openai` against a llama-swap model that runs `llama-server --embedding`. That model sits in a persistent group so chat swaps do not unload it.

### Ollama-compatible shim

llama-swap has no Ollama API. The System API adds `/api/tags`, `/api/show`, `/api/chat`, `/api/generate`, `/api/pull` and `/api/ps`. Each route translates to the llama-swap OpenAI API or to the Hugging Face download path. Caddy routes `/api/*` on port 8443 to the shim, and `/v1/*` to llama-swap.

## Host image

`os/Containerfile` starts from `quay.io/fedora/fedora-bootc:44` and adds:

- The packages in `os/packages/common.txt` and `os/packages/<variant>.txt`.
- The `neuraldrive` user, its subordinate ID ranges, and lingering so its user manager starts at boot.
- Quadlet units, the Podman network, sysctl and nftables rules.
- The container signature policy in `/etc/containers/policy.json`.
- Host-only tools that cannot run in a container: GPU detection, the installer and disk management.

The AMD variant needs no extra kernel modules. `amdgpu` is in the Fedora kernel and is signed for Secure Boot. The host needs only the firmware (`amd-gpu-firmware`) and device access. The ROCm userspace is in the HIP image only.

The NVIDIA variant adds `akmod-nvidia` built at image build time and signed with a project MOK key. The user enrolls that key once with `mokutil`. The Intel variant adds `intel-compute-runtime` to the host for device nodes, and the inference image uses Vulkan or SYCL.

### GPU access from rootless containers

The `neuraldrive` user is in the `render` and `video` groups. Quadlet passes `/dev/kfd` and `/dev/dri` with `AddDevice=` and keeps the groups with `GroupAdd=keep-groups`. SELinux blocks device access by `container_t` unless the `container_use_devices` boolean is on. `neuraldrive-selinux.service` sets it at each boot.

## Storage

| Mode | Root | Models and data | Unlock |
|---|---|---|---|
| Live USB | Read-only image from the USB | Persistence partition on the USB, label `ND_PERSIST`, mounted at `/var/lib/neuraldrive` | Passphrase at boot, optional |
| Installed | bootc on internal disk, A/B deployments | `/var` on the root disk, plus pools under `/var/mnt/models-N` | TPM2 through `systemd-cryptenroll`, passphrase as recovery |

The guided installer runs from the live USB. It wraps `bootc install to-disk` with these steps:

1. Pick a target disk. The installer lists disks with `lsblk -J` and excludes the boot USB.
2. Confirm the wipe. The installer prints the disk model, size and serial number first.
3. Create the LUKS2 root and install the image.
4. Enroll TPM2 with PCR 7 (Secure Boot state), and print a recovery key.
5. Copy models from the USB persistence partition, if the user asks.

Model storage pools are extra disks that the System API formats as LUKS2 plus XFS, with TPM2 unlock. Each pool mounts at `/var/mnt/models-N`. llama-swap sees all pools through one bind mount per pool. The TUI and System API show free space per pool.

The live USB is built with `bootc-image-builder --type iso`, plus a persistence partition that the first boot creates in the free space on the stick.

## Security

- SELinux stays enforcing. Only Arcane runs with `SecurityLabelDisable=true`.
- Secure Boot uses the Fedora shim and kernel. The NVIDIA variant adds the project MOK key.
- CI signs every image with cosign keyless signing. `/etc/containers/policy.json` requires a signature from this repo's workflow for `ghcr.io/ihavespoons/neuraldrive-*`. Third-party images are pinned by digest in the Quadlet files.
- Every Quadlet container drops all capabilities, sets `NoNewPrivileges=true` and `ReadOnly=true`, and gets a tmpfs for `/tmp`. Writable state goes to named volumes only.
- nftables default-deny. Only 443, 8443, 9443 and rate-limited 22 are open.
- SSH accepts keys only. Root login is off.
- `bootc` and Podman auto-update are off by default. Updates are explicit, from the TUI, the System API, or `bootc upgrade`.

## Phases

| Phase | Scope | Exit check |
|---|---|---|
| 1 | Inference images, host Containerfile, Quadlets, Caddy image, CI build and signing | The AMD host image boots in a VM. llama-swap serves a GGUF model on the 7900 XTX. |
| 2 | System API and TUI in containers. Replace `systemctl` calls with the Podman and systemd D-Bus APIs. Hugging Face picker. Ollama shim. Benchmark. | `tests/test_api.py` passes against the new stack. |
| 3 | Live ISO, persistence, guided installer, LUKS with TPM2, model pools | Install from USB to disk on the target host, and unlock without a passphrase. |
| 4 | NVIDIA and Intel variants, offline update bundles, signature policy enforced | Each variant boots. An unsigned image fails to pull. |
| 5 | Remove the live-build tree, update `AGENTS.md`, `README.md` and both mdbooks | No references to Ollama or live-build remain. |

## Open items

- The System API restarts services with `systemctl` (`config/includes.chroot/usr/lib/neuraldrive/api/neuraldrive_api/main.py`). In a container it must call the user systemd manager over D-Bus instead. Phase 2 must design that interface and limit it to the NeuralDrive units.
- The TUI runs on the console today. As a container it needs `/dev/tty1`. Phase 2 must decide between a container on tty1 and a host package.
- Arcane Git Sync polls and has no webhooks. Compose files pulled from Git are read-only in the Arcane UI.
