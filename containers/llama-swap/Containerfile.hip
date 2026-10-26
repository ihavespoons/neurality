# llama.cpp (HIP/ROCm) + llama-swap, built for one GPU target.
# A single-target build keeps the image small and the kernels tuned to the card.

ARG ROCM_VERSION=7.2.4
ARG ROCM_BUILD_IMAGE=docker.io/rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete
ARG RUNTIME_IMAGE=docker.io/ubuntu:24.04

FROM ${ROCM_BUILD_IMAGE} AS build

ARG LLAMA_CPP_REF=b11889
ARG GPU_TARGETS=gfx1100
ARG GGML_HIP_RCCL=OFF

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" https://github.com/ggml-org/llama.cpp /src

WORKDIR /src
RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIP=ON \
        -DGPU_TARGETS="${GPU_TARGETS}" \
        -DGGML_HIP_RCCL="${GGML_HIP_RCCL}" \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
    && cmake --build build --config Release -j"$(nproc)" --target llama-server llama-bench

# Collect the binaries, their shared libraries, and only the rocBLAS and
# hipBLASLt kernel files for GPU_TARGETS. The full ROCm tree is ~20 GB.
COPY collect-rocm-libs.sh /usr/local/bin/collect-rocm-libs.sh
RUN /usr/local/bin/collect-rocm-libs.sh /out "${GPU_TARGETS}" build/bin/llama-server build/bin/llama-bench

FROM ${RUNTIME_IMAGE} AS llama-swap-dl

ARG LLAMA_SWAP_VERSION=266
ARG LLAMA_SWAP_SHA256=
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY fetch-llama-swap.sh /usr/local/bin/fetch-llama-swap.sh
RUN /usr/local/bin/fetch-llama-swap.sh "${LLAMA_SWAP_VERSION}" "${LLAMA_SWAP_SHA256}" /out

FROM ${RUNTIME_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 libssl3t64 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --home /app --no-create-home app

COPY --from=build /out/ /
COPY --from=llama-swap-dl /out/llama-swap /app/llama-swap
COPY config.yaml /app/config.yaml

ENV PATH=/app:/opt/llama/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/llama/lib \
    ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library \
    HIPBLASLT_TENSILE_LIBPATH=/opt/rocm/lib/hipblaslt/library \
    SIBYL_GPU_BACKEND=ROCm

USER 10001:10001
WORKDIR /app
EXPOSE 8080
HEALTHCHECK CMD ["curl", "-fsS", "http://127.0.0.1:8080/health"]
ENTRYPOINT ["/app/llama-swap", "-listen", ":8080", "-config", "/app/config.yaml", "-watch-config"]
