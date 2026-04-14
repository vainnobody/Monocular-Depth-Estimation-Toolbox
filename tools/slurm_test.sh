#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

if [[ "$4" =~ ^[0-9]+$ ]]; then
    CHECKPOINT=""
    GPUS=$4
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
    PY_ARGS=${@:5}
else
    CHECKPOINT=$4
    PY_ARGS=${@:5}
fi

TEST_ARGS=("${CONFIG}")

if [[ -n "$CHECKPOINT" ]]; then
    TEST_ARGS+=("${CHECKPOINT}")
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/test.py "${TEST_ARGS[@]}" --launcher="slurm" ${PY_ARGS}
