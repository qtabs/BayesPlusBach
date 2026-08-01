#!/usr/bin/env bash
# Restart pipeline.py on failure, up to MAX_ATTEMPTS times with PAUSE seconds
# between tries.
#
# Why a shell loop and not a try/except inside Python: the crashes here are
# intermittent CUDA/cuDNN faults, and a CUDA error can leave the process's CUDA
# context corrupted -- retrying in the SAME process would just keep failing. A
# fresh interpreter each attempt gets a clean context. The pipeline is resumable
# (finished stages are checkpointed on disk and fit_or_load reloads them), so a
# restart continues from where the crash happened rather than redoing work.
#
# Usage:
#   ./run_pipeline.sh                # 100 attempts, 60s pause (defaults)
#   ./run_pipeline.sh 50 120         # 50 attempts, 120s pause
#   nohup ./run_pipeline.sh &        # detached; watch with: tail -f pipeline_run.log

set -uo pipefail

MAX_ATTEMPTS=${1:-100}
PAUSE=${2:-60}
LOG="pipeline_run.log"

for ((i=1; i<=MAX_ATTEMPTS; i++)); do
    echo "==== attempt $i/$MAX_ATTEMPTS  $(date '+%F %T') ====" | tee -a "$LOG"

    # PIPESTATUS[0] is python's exit code (tee, the last pipe stage, would mask it)
    python3 -u pipeline.py 2>&1 | tee -a "$LOG"
    code=${PIPESTATUS[0]}

    if [[ $code -eq 0 ]]; then
        echo "==== SUCCESS on attempt $i  $(date '+%F %T') ====" | tee -a "$LOG"
        exit 0
    fi

    echo "==== attempt $i FAILED (exit $code); sleeping ${PAUSE}s  $(date '+%F %T') ====" | tee -a "$LOG"
    sleep "$PAUSE"
done

echo "==== gave up after $MAX_ATTEMPTS attempts  $(date '+%F %T') ====" | tee -a "$LOG"
exit 1
