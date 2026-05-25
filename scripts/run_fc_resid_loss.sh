#!/bin/bash
# Residual-vs-persistence prediction sweep with selected event-weight settings.
# 6 configs x 3 holdouts x 3 seeds x 3 backbones = 162 runs.
# Configs are selected from the previous shared EventWeightedLoss sweep:
#   base
#   event_1h x {0.25, 0.5, 1.0}
#   event_decay x {0.25, 0.5}

set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate agc-mamba
cd ~/cast-greenhouse
export PYTHONPATH=.

mkdir -p results/runs results/logs
LOG=results/logs/fc_resid_loss_master.log
echo "[$(date)] START fc_resid_loss (residual prediction sweep, 162 runs; skip-if-exists)" \
    | tee -a "$LOG"

HOLDOUTS=(Reference Automatoes Digilog)
SEEDS=(42 0 123)
BACKBONES=(lstm transformer mamba)

CONFIGS=(
    "base:base:0"
    "e1h_l025:event_1h:0.25"
    "e1h_l050:event_1h:0.5"
    "e1h_l100:event_1h:1.0"
    "ed_l025:event_decay:0.25"
    "ed_l050:event_decay:0.5"
)
N_CFG=${#CONFIGS[@]}

launch_train() {
    local gpu=$1 holdout=$2 bb=$3 seed=$4 cfg=$5
    IFS=':' read -r tag loss lam <<< "$cfg"

    local tag_name=fc_resid_loss_${bb}_${tag}_h${holdout}_s${seed}
    local out_dir=results/runs/${tag_name}
    local logf=results/logs/${tag_name}.log

    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[skip] ${tag_name} (metrics.json exists)" | tee -a "$LOG"
        LAST_PID=""
        return
    fi

    local extra=""
    if [ "$loss" != "base" ]; then
        extra="--event-loss-lambda $lam"
    fi

    CUDA_VISIBLE_DEVICES=$gpu python scripts/train_one.py \
        --mode cross --holdout "$holdout" \
        --backbone "$bb" --seed "$seed" \
        --prediction-mode residual \
        --loss-mode "$loss" $extra \
        --run-name "$tag_name" \
        --quiet \
        > "$logf" 2>&1 &
    LAST_PID=$!
}

run_pair() {
    local holdout=$1 bb=$2 seed=$3 c1=$4 c2=$5

    LAST_PID=""
    launch_train 0 "$holdout" "$bb" "$seed" "$c1"
    local pid_a=$LAST_PID

    local pid_b=""
    if [ -n "$c2" ]; then
        LAST_PID=""
        launch_train 1 "$holdout" "$bb" "$seed" "$c2"
        pid_b=$LAST_PID
    fi

    [ -n "$pid_a" ] && wait "$pid_a" || true
    [ -n "$pid_b" ] && wait "$pid_b" || true
}

for holdout in "${HOLDOUTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for bb in "${BACKBONES[@]}"; do
            echo "[$(date)] starting holdout=$holdout seed=$seed bb=$bb (6 configs)" \
                | tee -a "$LOG"

            for ((i=0; i < N_CFG; i+=2)); do
                c1=${CONFIGS[$i]}
                c2=${CONFIGS[$((i+1))]:-}
                run_pair "$holdout" "$bb" "$seed" "$c1" "$c2"
            done

            echo "[$(date)] done holdout=$holdout seed=$seed bb=$bb" | tee -a "$LOG"
        done
    done
done

echo "[$(date)] ALL TRAIN DONE" | tee -a "$LOG"

echo "[$(date)] event-window eval" | tee -a "$LOG"
python scripts/evaluate_event_windows.py \
    --run-glob 'fc_resid_loss_*' \
    --windows 1h 3h 6h \
    --out-csv results/fc_resid_loss_event_windows.csv 2>&1 \
    | tee results/logs/fc_resid_loss_eval_windows.log

echo "[$(date)] event-timing eval" | tee -a "$LOG"
python scripts/evaluate_event_timing.py \
    --run-glob 'fc_resid_loss_*' \
    --windows 1h 3h 6h \
    --response-fraction 0.5 \
    --out-csv results/fc_resid_loss_event_timing.csv 2>&1 \
    | tee results/logs/fc_resid_loss_eval_timing.log

echo "[$(date)] alpha export" | tee -a "$LOG"
python scripts/export_alpha.py \
    --run-glob 'fc_resid_loss_*' \
    --splits test 2>&1 | tee results/logs/fc_resid_loss_alpha_export.log

echo "[$(date)] ALL DONE" | tee -a "$LOG"
