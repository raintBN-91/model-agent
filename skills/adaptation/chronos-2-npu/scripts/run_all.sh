#!/bin/bash
# =============================================================================
# Chronos-2 Ascend NPU One-Click Evaluation Pipeline
# =============================================================================
# Usage:
#   bash run_all.sh                          # Full evaluation
#   bash run_all.sh --quick                  # Quick test (fewer runs)
#   bash run_all.sh --skip-context           # Skip context scaling test
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/tmp/chronos2_model/amazon/chronos-2}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/evaluation}"
LOGS_DIR="${SCRIPT_DIR}/logs"
SCREENSHOTS_DIR="${SCRIPT_DIR}/screenshots"

MODE="${1:-full}"
NUM_RUNS=100

case "$MODE" in
    --quick)
        NUM_RUNS=20
        EXTRA_ARGS=""
        echo ">>> QUICK MODE: ${NUM_RUNS} runs per test <<<"
        ;;
    --skip-context)
        NUM_RUNS=100
        EXTRA_ARGS="--skip_context_scale"
        echo ">>> Skipping context length scaling <<<"
        ;;
    *)
        EXTRA_ARGS=""
        echo ">>> FULL EVALUATION: ${NUM_RUNS} runs per test <<<"
        ;;
esac

# Ensure output dirs
mkdir -p "${LOGS_DIR}" "${SCREENSHOTS_DIR}" "${OUTPUT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOGS_DIR}/evaluation_${TIMESTAMP}.log"

# Source CANN environment if available
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# Set NPU optimizations
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=0

echo "============================================"
echo " Chronos-2 Ascend NPU Evaluation Pipeline"
echo " Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Model: ${MODEL_PATH}"
echo " Output: ${OUTPUT_DIR}"
echo " Log: ${LOG_FILE}"
echo "============================================"

# Step 1: Download model if not exists
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[Step 1] Downloading model weights ..."
    python3 "${SCRIPT_DIR}/download_model.py"
else
    echo "[Step 1] Model found at ${MODEL_PATH}"
fi

# Step 2: Quick smoke test
echo "[Step 2] NPU Smoke Test ..."
python3 -c "
import os, warnings; warnings.filterwarnings('ignore')
os.environ['TASK_QUEUE_ENABLE'] = '1'
import torch, torch_npu
from chronos import Chronos2Pipeline
p = Chronos2Pipeline.from_pretrained('${MODEL_PATH}', dtype=torch.float32, device_map=None)
p.model = p.model.to('npu:0'); p.model.eval()
inputs = torch.randn(1, 1, 512)
for _ in range(5): _ = p.predict(inputs, prediction_length=64)
torch.npu.synchronize()
import time; t0 = time.time()
for _ in range(10):
    _ = p.predict(inputs, prediction_length=64)
    torch.npu.synchronize()
print(f'NPU Smoke Test PASSED: avg {(time.time()-t0)/10*1000:.1f}ms')
" 2>&1 | tee -a "${LOG_FILE}"

# Step 3: Full evaluation
echo ""
echo "[Step 3] Running Full Evaluation Suite ..."
python3 "${SCRIPT_DIR}/evaluate.py" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_runs "${NUM_RUNS}" \
    ${EXTRA_ARGS} \
    2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "============================================"
echo " Evaluation Complete!"
echo " Reports: ${OUTPUT_DIR}/"
echo " Log:     ${LOG_FILE}"
echo "============================================"
