#!/bin/sh
# Download a llama-swap release binary and check it against the release checksums.
# Usage: fetch-llama-swap.sh <version> <sha256-or-empty> <outdir>
set -eu

version="$1"
expected="$2"
out="$3"
base="https://github.com/mostlygeek/llama-swap/releases/download/v${version}"
tarball="llama-swap_${version}_linux_amd64.tar.gz"

mkdir -p "$out"
cd "$out"
curl -fsSLO "${base}/${tarball}"
curl -fsSLO "${base}/llama-swap_${version}_checksums.txt"

if [ -n "$expected" ]; then
    echo "${expected}  ${tarball}" | sha256sum -c -
else
    grep " ${tarball}\$" "llama-swap_${version}_checksums.txt" | sha256sum -c -
fi

tar -xzf "$tarball" llama-swap
rm -f "$tarball" "llama-swap_${version}_checksums.txt"
