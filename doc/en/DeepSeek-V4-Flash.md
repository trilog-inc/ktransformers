# DeepSeek-V4-Flash-0731 DSpark on RTX PRO 6000

This recipe runs `deepseek-ai/DeepSeek-V4-Flash-0731` on one 96 GB RTX PRO
6000 (SM120) with SGLang DSpark and KT-Kernel CPU/GPU expert offload. CPU
experts remain in the checkpoint's native MXFP4 representation: packed E2M1
weights with one UE8M0 scale per 32 values. There is no AMXINT4 conversion.

## Execution split

- The target's hot routed experts use SGLang's `flashinfer_mxfp4` SM120 path.
- The target's remaining routed experts use KT-Kernel. Multi-token expert
  batches use AMX-BF16 tiles; small batches use the AVX512-BF16 tail kernel.
- KT assigns the CPU complement compact IDs and loads only those logical
  experts, so GPU-resident routed weights are not duplicated in KT buffers.
- The bundled DSpark/MTP layer stays entirely on the GPU. It is loaded from the
  same 0731 checkpoint and is not duplicated in CPU memory.
- KT runs as an eager node inside SGLang's breakable decode CUDA graph. The GPU
  graph segments around every target MoE layer remain captured.

The AMX implementation expands one 32-value MXFP4 tile into a small BF16 VNNI
scratch tile immediately before the dot product. It never creates a BF16 copy
of the model, never treats E2M1 nibbles as integer weights, and applies the
original per-output-channel scale in FP32 after each K=32 partial.

## Requirements

- Linux x86-64 CPU exposing `amx_tile`, `amx_bf16`, AVX512F/BW/VL, and
  AVX512-BF16.
- One RTX PRO 6000 Blackwell GPU with compute capability 12.0 and 96 GB VRAM.
- CUDA Toolkit 13.3 and an NVIDIA driver compatible with that toolkit.
- Enough system RAM for the native checkpoint and KT's CPU expert buffers.
  256 GB is a lower bound; 384 GB or more leaves substantially better file
  cache and NUMA headroom.
- Current SGLang main containing DSpark support and FlashInfer with SM120
  MXFP4 support.

Verify the host before building:

```bash
/usr/local/cuda-13.3/bin/nvcc --version
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E 'amx_tile|amx_bf16|avx512_bf16'
```

## Build KT-Kernel

```bash
git submodule update --init --recursive
cd kt-kernel

export CUDA_HOME=/usr/local/cuda-13.3
export CPUINFER_CPU_INSTRUCT=NATIVE
export CPUINFER_ENABLE_AMX=ON
export CPUINFER_ENABLE_AVX512=ON
export CPUINFER_ENABLE_AVX512_BF16=ON
export CPUINFER_USE_CUDA=1
export CPUINFER_CUDA_ARCHS=120

python3 -m pip install --no-build-isolation -v .
```

`-mamx-int8` may still appear in the extension's aggregate compiler flags
because other pre-existing KT operators use it. The MXFP4 kernel described here
uses `_tile_dpbf16ps`; it contains no AMX-INT4 or integer dot-product path.

The compiled module exposes `kt_kernel_ext.moe.HAS_AMX_BF16`. With
`KT_MXFP4_BACKEND=amx`, startup fails instead of silently falling back if either
the binary or host lacks AMX-BF16.

## Download the checkpoint

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /models/DeepSeek-V4-Flash-0731
```

Point both `--model-path` and `--kt-weight-path` at that directory. No weight
conversion step is required. KT reads keys such as
`layers.0.ffn.experts.0.w1.weight` and `.scale` directly from the 48-shard
checkpoint.

## Launch

```bash
export CUDA_HOME=/usr/local/cuda-13.3
export FLASHINFER_CUDA_ARCH_LIST=12.0a
export TORCH_CUDA_ARCH_LIST="12.0+PTX"

python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path /models/DeepSeek-V4-Flash-0731 \
  --tp 1 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algorithm DSPARK \
  --kt-weight-path /models/DeepSeek-V4-Flash-0731 \
  --kt-method MXFP4 \
  --kt-mxfp4-backend amx \
  --kt-mxfp4-amx-min-tokens-per-expert 4 \
  --kt-num-gpu-experts 96 \
  --kt-expert-placement-strategy uniform \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-numa-nodes 0 1 \
  --disable-shared-experts-fusion \
  --mem-fraction-static 0.86 \
  --chunked-prefill-size 4096 \
  --swa-full-tokens-ratio 0.1 \
  --host 0.0.0.0 \
  --port 30000
```

Do not pass `--speculative-draft-model-path`: the 0731 checkpoint declares one
DSpark layer, block size 5, and target capture layers 40–42. SGLang reads those
values from `config.json`. SGLang automatically selects a breakable decode
graph for the KT target while retaining the full-graph fast path for the
GPU-only draft. Passing an explicit breakable backend remains supported but
also applies that choice to the draft.

The SGLang tree also provides
`scripts/launch_dsv4_flash_0731_dspark_kt_sm120.sh` with `check`, `build-kt`,
and `serve` actions.

## Performance tuning

1. Start with 96 GPU experts per layer. Reduce it in steps of 8 if model load,
   FlashInfer post-processing, KV allocation, or graph capture exceeds 96 GB.
   Increase it if VRAM remains and the CPU phase is the layer critical path.
2. Profile `--kt-mxfp4-amx-min-tokens-per-expert` at 0, 2, 4, 8, and 16.
   Decode at batch one often routes only one token to an expert, where AVX512
   avoids AMX padding. DSpark verification and concurrent requests create the
   larger expert batches where AMX wins.
3. Collect routed-expert counts under representative traffic and use
   `--kt-expert-placement-strategy frequency --init-expert-location counts.pt`.
   This normally beats a uniform mask once the workload is stable.
4. Use one KT thread pool per populated CPU NUMA node and list the node IDs
   explicitly. Keep the GPU and its PCIe root complex close to the pool that
   handles staging when the platform topology permits it.
5. Compare DSpark block sizes under the real concurrency distribution. The
   checkpoint default proposes five tokens. More verification work increases
   both GPU and CPU expert batch sizes, so acceptance length and end-to-end TPOT
   matter more than AMX utilization by itself.
6. Tune `--mem-fraction-static` and `--chunked-prefill-size` together. Leave
   several GiB outside SGLang for FlashInfer JIT/autotuning and graph capture.

For correctness, first compare target-only and DSpark output distributions with
the same greedy prompts. Then benchmark warmed servers with identical request
sets, cache state, prompt lengths, output lengths, and concurrency. Track TTFT,
TPOT, accepted tokens per DSpark step, GPU memory, CPU memory bandwidth, AMX
utilization, and the overlap between CPU and GPU MoE phases.
