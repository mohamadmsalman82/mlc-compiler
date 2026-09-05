#!/usr/bin/env bash
# The benchmark sweep. Run only after gpu_check.sh passes.
#
# BERT and GPT are run separately: GPT-2's lm_head produces a 785 MB output
# tensor at batch 32, which crowds a small card and dominates its timing, so
# it stops at batch 8 unless the card is large.
set -o pipefail
cd "$(dirname "$0")/.."
mkdir -p results
exec > >(tee results/bench.log) 2>&1

VRAM=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory/1024**3))")
GPT_BATCHES="1 8"
[ "$VRAM" -ge 20 ] && GPT_BATCHES="1 8 32"
echo "detected ${VRAM} GiB, running gpt2-small at batches: ${GPT_BATCHES}"

python -m mlc.bench --calibrate --dtype float16 \
    --models bert-base --batches 1 8 32 \
    --out results/bert-fp16.md || exit 1

python -m mlc.bench --calibrate --dtype float16 \
    --models gpt2-small --batches $GPT_BATCHES \
    --out results/gpt-fp16.md || exit 1

python -m mlc.bench --calibrate --dtype float32 \
    --models bert-base --batches 1 8 \
    --out results/bert-fp32.md || exit 1

echo
echo "=============== done ==============="
ls -la results/
echo "send back everything in results/"
