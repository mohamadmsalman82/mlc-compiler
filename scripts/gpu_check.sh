#!/usr/bin/env bash
# First run on a GPU. Installs, tests, compiles and executes three models
# through Triton, then measures the cost model's constants.
#
# Everything is logged to gpu_check.log. If anything fails, that file is what
# to send back: it has the versions, the failing traceback, and the state of
# every step before it.
set -o pipefail
cd "$(dirname "$0")/.."
exec > >(tee gpu_check.log) 2>&1

step() { echo; echo "=============== $* ==============="; echo; }

step "environment"
nvidia-smi || { echo "FAIL: no GPU visible"; exit 1; }
python -c "import sys; print('python', sys.version)"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, "| sm", f"{p.major}.{p.minor}",
          "| vram", round(p.total_memory / 1024**3, 1), "GiB")
try:
    import triton; print("triton", triton.__version__)
except ImportError:
    print("triton NOT IMPORTABLE -- the Triton backend cannot run")
PY

step "test suite (does not touch the GPU)"
python -m pytest tests/ -q || { echo "FAIL: tests"; exit 1; }

step "smallest model through Triton"
python -m mlc run gpt-small --device cuda --dtype float16 || exit 1

step "full-size models"
python -m mlc run bert-base  --device cuda --dtype float16            || exit 1
python -m mlc run gpt2-small --device cuda --dtype float16            || exit 1

step "cuda graph capture at batch 32"
python -m mlc run bert-base --device cuda --dtype float16 --batch 32  || exit 1

step "streamed reduction path"
python -m mlc run bert-base --device cuda --dtype float16 --max-row 64 || exit 1

step "fp32, for comparison"
python -m mlc run bert-base --device cuda --dtype float32             || exit 1

step "calibration"
python -m mlc.bench --calibrate --dtype float16 --models gpt-small \
    --batches 1 --iters 20 --variants eager mlc/+memory

echo
echo "=============== all checks passed ==============="
echo "send back gpu_check.log, then run scripts/gpu_bench.sh"
