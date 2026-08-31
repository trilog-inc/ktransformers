# GLM-5.2 NVFP4 on SM120 with KT-Kernel

This experimental path serves the serialized `nvidia/GLM-5.2-NVFP4`
checkpoint on an SM120 GPU while keeping non-resident routed experts in their
native ModelOpt NVFP4 representation on CPU.

The CPU backend consumes the checkpoint tensors directly:

- packed E2M1 weights, two values per byte;
- one E4M3 block scale per 16 input values;
- independent FP32 `weight_scale_2` values for gate, up, and down projections.

CPU expert residency is approximately 0.5625 bytes per parameter, excluding
allocator metadata. Activations remain BF16. The FP32 projection scale is
applied after accumulation, so checkpoint activation scales are not applied a
second time.

## Update Both Repositories

Set the paths to the existing clones on the inference server:

```bash
export SGLANG_DIR=/mnt/home_extend/llm/ktglm52/sglang
export KTRANSFORMERS_DIR=/mnt/home_extend/llm/ktglm52/ktransformers
export BRANCH=codex/glm5-nextn-mtp-4090

git -C "$SGLANG_DIR" fetch myrepo "$BRANCH"
git -C "$SGLANG_DIR" checkout "$BRANCH"
git -C "$SGLANG_DIR" pull --ff-only myrepo "$BRANCH"

git -C "$KTRANSFORMERS_DIR" fetch trilog "$BRANCH"
git -C "$KTRANSFORMERS_DIR" checkout "$BRANCH"
git -C "$KTRANSFORMERS_DIR" pull --ff-only trilog "$BRANCH"
git -C "$KTRANSFORMERS_DIR" submodule update --init --recursive
```

Use the actual remote names from `git remote -v` if the server clones use
different names.

## Rebuild KT-Kernel

Build for the inference CPU rather than producing a portable binary. The
AVX-512 BF16 backend is preferred when available; AVX2 is the functional
fallback.

```bash
conda activate ktglm52
cd "$KTRANSFORMERS_DIR/kt-kernel"

export CPUINFER_CPU_INSTRUCT=NATIVE
export CPUINFER_ENABLE_AMX=ON
export CPUINFER_BUILD_TYPE=Release
export CPUINFER_FORCE_REBUILD=1
export CPUINFER_PARALLEL="$(nproc)"

python -m pip install -v \
  --force-reinstall \
  --no-build-isolation \
  --no-cache-dir \
  --no-deps \
  .

python -m pip install -e "$SGLANG_DIR/python" --no-deps
```

Verify that the native backend is present:

```bash
python - <<'PY'
from kt_kernel import kt_kernel_ext

available = [
    name
    for name in ("AMXFP4_KGroup_MOE", "AVX2MXFP4_MOE")
    if hasattr(kt_kernel_ext.moe, name)
]
print("NVFP4 CPU backends:", available)
assert available
PY
```

## Download the Checkpoint

```bash
export MODEL_PATH=/mnt/home_extend/models/GLM-5.2-NVFP4

hf download nvidia/GLM-5.2-NVFP4 \
  --local-dir "$MODEL_PATH"
```

Do not point `--kt-weight-path` at the FP8 checkpoint. The server now rejects a
ModelOpt NVFP4 GPU method paired with a non-NVFP4 KT method, and vice versa.

## Launch on RTX PRO 6000 Blackwell

This is the performance-oriented TP1 baseline for a 96 GB SM120 GPU. Adjust
CPU threads, NUMA nodes, and resident expert count for the host. Start with a
smaller `KT_GPU_EXPERTS` value if GPU memory is tight.

```bash
conda activate ktglm52
cd "$SGLANG_DIR"

export MODEL_PATH=/mnt/home_extend/models/GLM-5.2-NVFP4
export KT_CPU_THREADS=26
export KT_GPU_EXPERTS=30
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_JIT_DEEPGEMM=0

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --kt-weight-path "$MODEL_PATH" \
  --kt-method NVFP4 \
  --kt-cpuinfer "$KT_CPU_THREADS" \
  --kt-threadpool-count 2 \
  --kt-numa-nodes 0 1 \
  --kt-num-gpu-experts "$KT_GPU_EXPERTS" \
  --kt-gpu-prefill-token-threshold 0 \
  --kt-expert-placement-strategy uniform \
  --tp-size 1 \
  --moe-runner-backend flashinfer_cutlass \
  --fp4-gemm-backend flashinfer_cutlass \
  --attention-backend nsa \
  --nsa-prefill-backend flashinfer_sparse_mla \
  --nsa-decode-backend flashinfer_sparse_mla \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --disable-shared-experts-fusion \
  --mem-fraction-static 0.95 \
  --max-total-tokens 4096 \
  --max-running-requests 2 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --served-model-name GLM5.2-NVFP4 \
  --host 0.0.0.0 \
  --port 8000
```

Native NVFP4 currently requires static expert residency. Do not add
`--kt-enable-dynamic-expert-update`, and keep
`--kt-gpu-prefill-token-threshold 0`.

The SM120 sparse MLA path requires a FlashInfer build that provides
`flashinfer_sparse_mla` support for the installed CUDA and PyTorch versions.
Do not use the SM90/SM100 FlashMLA metadata path on SM120.

## Smoke Test

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM5.2-NVFP4",
    "messages": [{"role": "user", "content": "What is 17 times 23?"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

For performance measurement, warm the server first and compare prompt and
decode throughput separately. Record `KT_GPU_EXPERTS`, CPU affinity, NUMA
placement, prompt length, concurrency, and whether CUDA graphs were captured.
