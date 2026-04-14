#!/usr/bin/env bash

CONFIG=$1

if [[ "$2" =~ ^[0-9]+$ ]]; then
    CHECKPOINT=""
    GPUS=$2
    PY_ARGS=${@:3}
else
    CHECKPOINT=$2
    GPUS=$3
    PY_ARGS=${@:4}
fi

PORT=${PORT:-29547}
TEST_ARGS=("$CONFIG")

if [[ -n "$CHECKPOINT" ]]; then
    TEST_ARGS+=("$CHECKPOINT")
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py "${TEST_ARGS[@]}" --launcher pytorch ${PY_ARGS}
