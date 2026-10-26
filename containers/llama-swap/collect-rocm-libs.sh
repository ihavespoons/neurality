#!/bin/bash
# Copy binaries plus every shared library they load into <outdir>, and keep only
# the rocBLAS / hipBLASLt kernel files that match the GPU targets.
# Usage: collect-rocm-libs.sh <outdir> <gpu-targets;sep> <binary>...
set -euo pipefail

out="$1"
targets="$2"
shift 2

rocm="$(hipconfig -R)"
mkdir -p "$out/opt/llama/bin" "$out/opt/llama/lib" "$out/opt/rocm/lib"

copy_lib() {
    local lib="$1" real dest
    real="$(readlink -f "$lib")"
    case "$real" in
        "$rocm"/*) dest="$out/opt/rocm/lib" ;;
        /src/*)    dest="$out/opt/llama/lib" ;;
        *)         return 0 ;;   # system libraries come from the runtime image
    esac
    cp -n "$real" "$dest/"
    [ "$(basename "$lib")" = "$(basename "$real")" ] || ln -sf "$(basename "$real")" "$dest/$(basename "$lib")"
}

for bin in "$@"; do
    cp "$bin" "$out/opt/llama/bin/"
    ldd "$bin" | awk '/=> \//{print $3}' | while read -r lib; do copy_lib "$lib"; done
done

# Libraries that ROCm loads with dlopen, so ldd does not list them.
for lib in "$rocm"/lib/libamd_comgr.so* "$rocm"/lib/libhsa-runtime64.so* "$rocm"/lib/librocprofiler-register.so*; do
    [ -e "$lib" ] && copy_lib "$lib"
done

IFS=';' read -ra archs <<< "$targets"
for kdir in rocblas/library hipblaslt/library; do
    [ -d "$rocm/lib/$kdir" ] || continue
    mkdir -p "$out/opt/rocm/lib/$kdir"
    for arch in "${archs[@]}"; do
        find "$rocm/lib/$kdir" -maxdepth 1 -name "*${arch}*" -exec cp -P {} "$out/opt/rocm/lib/$kdir/" \;
    done
    # Architecture-independent index files.
    find "$rocm/lib/$kdir" -maxdepth 1 -type f ! -name '*gfx*' -exec cp -P {} "$out/opt/rocm/lib/$kdir/" \;
done

du -sh "$out"
