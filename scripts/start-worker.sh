#!/bin/bash
# =============================================================================
# start-worker.sh — Celery worker entrypoint with auto-tuned CPU allocation
#
# Reads the total core count at runtime and derives sensible concurrency and
# OMP_NUM_THREADS values so simulations never starve the LLM or other services.
#
# Env vars (all optional — defaults give a balanced split on any machine):
#
#   SIMULATION_CPU_FRACTION   Fraction of total cores reserved for FEBio jobs.
#                             Default: 0.5  (50% for sims, 50% for everything else)
#                             Increase on machines used only for batch simulation.
#                             Decrease if the LLM feels sluggish during a run.
#
#   WORKER_CONCURRENCY        Number of parallel FEBio jobs.
#                             Default: auto — computed as floor(sim_cores / 8).
#                             Override with an integer to pin the value.
#
#   OMP_NUM_THREADS           Threads per FEBio process.
#                             Default: auto — computed as floor(sim_cores / concurrency).
#                             Override to pin OpenMP threads regardless of core count.
#
# Examples (for a 64-core machine):
#   Default (fraction=0.5):  sim_cores=32, concurrency=4, omp=8
#   Heavy sim workload:      SIMULATION_CPU_FRACTION=0.75 → sim_cores=48, concurrency=6, omp=8
#   Single-job deep run:     WORKER_CONCURRENCY=1 → concurrency=1, omp=32
# =============================================================================

set -e

TOTAL_CORES=$(nproc)
FRACTION=${SIMULATION_CPU_FRACTION:-0.5}
CONCURRENCY_OVERRIDE=${WORKER_CONCURRENCY:-auto}
OMP_OVERRIDE=${OMP_NUM_THREADS:-auto}

# ---------------------------------------------------------------------------
# Compute sim_cores — integer, at least 1
# ---------------------------------------------------------------------------
SIM_CORES=$(python3 -c "import math; print(max(1, math.floor(${TOTAL_CORES} * ${FRACTION})))")

# ---------------------------------------------------------------------------
# Compute concurrency — floor(sim_cores / 8), at least 1
# One job per 8 cores leaves headroom within each FEBio process for OpenMP.
# ---------------------------------------------------------------------------
if [ "${CONCURRENCY_OVERRIDE}" = "auto" ]; then
    CONCURRENCY=$(python3 -c "import math; print(max(1, math.floor(${SIM_CORES} / 8)))")
else
    CONCURRENCY=${CONCURRENCY_OVERRIDE}
fi

# ---------------------------------------------------------------------------
# Compute OMP_NUM_THREADS — floor(sim_cores / concurrency), at least 1
# ---------------------------------------------------------------------------
if [ "${OMP_OVERRIDE}" = "auto" ]; then
    OMP=$(python3 -c "import math; print(max(1, math.floor(${SIM_CORES} / ${CONCURRENCY})))")
else
    OMP=${OMP_OVERRIDE}
fi

export OMP_NUM_THREADS=${OMP}

echo "========================================================"
echo " Worker CPU allocation"
echo "  Total cores available : ${TOTAL_CORES}"
echo "  Simulation fraction   : ${FRACTION}"
echo "  Cores for simulations : ${SIM_CORES}"
echo "  Parallel FEBio jobs   : ${CONCURRENCY}  (WORKER_CONCURRENCY)"
echo "  Threads per FEBio job : ${OMP}           (OMP_NUM_THREADS)"
echo "  Cores left for stack  : $((TOTAL_CORES - SIM_CORES))  (LLM + API + services)"
echo "========================================================"

exec celery \
    --app digital_twin_ui.tasks.celery_app:celery_app \
    worker \
    --loglevel=info \
    --concurrency=${CONCURRENCY} \
    --queues=celery
