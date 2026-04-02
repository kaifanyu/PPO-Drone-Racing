#!/usr/bin/env bash
# =============================================================================
# fine_tune_sweep.sh
# Grid-search fine-tune from checkpoint over velocity / RP penalty / lookahead.
#
# Parameters swept:
#   VEL_SCALES      – velocity_toward_gate_reward_scale
#   RP_MULTIPLIERS  – multiplier on base roll/pitch penalty (-0.8)
#   NEXT_GATE_WEIGHTS – next-gate lookahead blend weight (code-level patch)
#
# Total runs: 3 × 2 × 2 = 12 × 1000 iters each
#
# Usage:
#   chmod +x fine_tune_sweep.sh
#   ./fine_tune_sweep.sh
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_SCRIPT="scripts/rsl_rl/train_race.py"
STRATEGY_FILE="/home/ohossain/Documents/Repos/ESE651/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py"   # <-- set this
TRAIN_CONFIG="/home/ohossain/Documents/Repos/ESE651/ese651_project/scripts/rsl_rl/train_race.py"             # <-- set this (file with rewards dict)

# ── Base checkpoint ──────────────────────────────────────────────────────────
BASE_RUN="2026-04-02_10-42-48"
BASE_CHECKPOINT="best_model.pt"

# ── Sweep grid ───────────────────────────────────────────────────────────────
VEL_SCALES=(8.0 15.0 30.0)
RATE_PENALTIES=(0.0 -0.16)           # applied to base -0.8  →  0.0 or -0.16
NEXT_GATE_WEIGHTS=(0.20 0.35)     # weight on next_gate_dir_unit in blended_dir

# ── Fixed training args ──────────────────────────────────────────────────────
TASK="Isaac-Quadcopter-Race-v0"
NUM_ENVS=2048
MAX_ITER=20
LOGGER="wandb"
BASE_RP=-0.8                      # your current roll_pitch_rates_reward_scale

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR="sweep_logs"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/sweep_summary.csv"
echo "run_name,vel_scale,rp_mult,rp_value,next_gate_w,status,start_time,end_time" > "$SUMMARY"

# ── Helpers ──────────────────────────────────────────────────────────────────

# Backup a file once; if backup already exists, skip (preserves original)
backup_file() {
    local f="$1"
    [[ ! -f "${f}.orig" ]] && cp "$f" "${f}.orig"
}

# Restore original from backup
restore_file() {
    local f="$1"
    [[ -f "${f}.orig" ]] && cp "${f}.orig" "$f"
}

# Patch velocity_toward_gate_reward_scale in the rewards dict
patch_velocity_scale() {
    local file="$1" val="$2"
    # Matches:  'velocity_toward_gate_reward_scale': <number>,
    sed -i "s/'velocity_toward_gate_reward_scale':[ ]*[0-9.]*/'velocity_toward_gate_reward_scale': ${val}/" "$file"
}

# Patch roll_pitch_rates_reward_scale in the rewards dict
patch_rate_penalties() {
    local file="$1" val="$2"
    sed -i "s/roll_pitch_rates_reward_scale = [-0-9.]*/roll_pitch_rates_reward_scale = ${val}/" "$file"
    sed -i "s/yaw_rate_reward_scale = [-0-9.]*/yaw_rate_reward_scale = ${val}/" "$file"
}

# Patch the blended lookahead weights in the strategy file
# Original line:  blended_dir = 0.8 * gate_dir_unit + 0.2 * next_gate_dir_unit
patch_lookahead() {
    local file="$1" next_w="$2"
    local curr_w
    curr_w=$(python3 -c "print(round(1.0 - ${next_w}, 4))")
    sed -i "s/blended_dir = [0-9.]* \* gate_dir_unit + [0-9.]* \* next_gate_dir_unit/blended_dir = ${curr_w} * gate_dir_unit + ${next_w} * next_gate_dir_unit/" "$file"
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "=== Pre-flight check ==="
[[ -f "$TRAIN_SCRIPT" ]]   || { echo "ERROR: TRAIN_SCRIPT not found: $TRAIN_SCRIPT"; exit 1; }
[[ -f "$STRATEGY_FILE" ]]  || { echo "ERROR: STRATEGY_FILE not found: $STRATEGY_FILE"; exit 1; }
[[ -f "$TRAIN_CONFIG" ]]   || { echo "ERROR: TRAIN_CONFIG not found: $TRAIN_CONFIG"; exit 1; }

# Verify the patterns we're about to sed actually exist
grep -q "velocity_toward_gate_reward_scale" "$TRAIN_CONFIG"  || { echo "ERROR: velocity pattern not found in $TRAIN_CONFIG"; exit 1; }
grep -q "roll_pitch_rates_reward_scale ="   "$TRAIN_CONFIG"  || { echo "ERROR: rp pattern not found in $TRAIN_CONFIG"; exit 1; }
grep -q "blended_dir = "                    "$STRATEGY_FILE" || { echo "ERROR: blended_dir pattern not found in $STRATEGY_FILE"; exit 1; }

echo "All files and patterns verified."

# Take initial backups
backup_file "$STRATEGY_FILE"
backup_file "$TRAIN_CONFIG"

# ── Trap: always restore originals on exit / error ────────────────────────────
cleanup() {
    echo ""
    echo "=== Restoring original files ==="
    restore_file "$STRATEGY_FILE"
    restore_file "$TRAIN_CONFIG"
    echo "Originals restored."
}
trap cleanup EXIT

# ── Run counter ───────────────────────────────────────────────────────────────
TOTAL=$(( ${#VEL_SCALES[@]} * ${#RP_MULTIPLIERS[@]} * ${#NEXT_GATE_WEIGHTS[@]} ))
RUN_NUM=0

echo ""
echo "=== Starting sweep: ${TOTAL} runs × ${MAX_ITER} iterations each ==="
echo ""
for VEL in "${VEL_SCALES[@]}"; do
    for RATE_PEN in "${RATE_PENALTIES[@]}"; do
        for NGW in "${NEXT_GATE_WEIGHTS[@]}"; do

            RUN_NUM=$(( RUN_NUM + 1 ))
            RUN_NAME="ft_vel${VEL}_rates${RATE_PEN}_ng${NGW}"
            RUN_LOG="$LOG_DIR/${RUN_NAME}.log"

            echo "────────────────────────────────────────────────────────────"
            echo " Run ${RUN_NUM}/${TOTAL}: ${RUN_NAME}"
            echo "   velocity_scale          = ${VEL}"
            echo "   roll_pitch_rates_scale  = ${RATE_PEN}"
            echo "   yaw_rate_scale          = ${RATE_PEN}"
            echo "   next_gate_weight        = ${NGW}"
            echo "────────────────────────────────────────────────────────────"

            cp "${STRATEGY_FILE}.orig" "$STRATEGY_FILE"
            cp "${TRAIN_CONFIG}.orig"  "$TRAIN_CONFIG"

            patch_velocity_scale  "$TRAIN_CONFIG"  "$VEL"
            patch_rate_penalties  "$TRAIN_CONFIG"  "$RATE_PEN"   # <-- replaces both in one call
            patch_lookahead       "$STRATEGY_FILE" "$NGW"

            echo "  [patch check] $(grep 'velocity_toward_gate_reward_scale' "$TRAIN_CONFIG")"
            echo "  [patch check] $(grep 'roll_pitch_rates_reward_scale ='    "$TRAIN_CONFIG")"
            echo "  [patch check] $(grep 'yaw_rate_reward_scale ='            "$TRAIN_CONFIG")"
            echo "  [patch check] $(grep 'blended_dir ='                      "$STRATEGY_FILE")"

            START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
            STATUS="success"

            python "$TRAIN_SCRIPT" \
                --task           "$TASK"            \
                --num_envs       "$NUM_ENVS"        \
                --max_iterations "$MAX_ITER"        \
                --headless                          \
                --logger         "$LOGGER"          \
                --resume         True               \
                --load_run       "$BASE_RUN"        \
                --checkpoint     "$BASE_CHECKPOINT" \
                --run_name       "$RUN_NAME"        \
                2>&1 | tee "$RUN_LOG" \
            || STATUS="FAILED"

            END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
            echo "${RUN_NAME},${VEL},${RATE_PEN},${RATE_PEN},${NGW},${STATUS},${START_TIME},${END_TIME}" >> "$SUMMARY"

            [[ "$STATUS" == "FAILED" ]] \
                && echo "  !! Run ${RUN_NAME} FAILED. Check ${RUN_LOG}" \
                || echo "  ✓  Run ${RUN_NAME} complete."
            echo ""
        done
    done
done

echo "============================================================"
echo " Sweep complete. ${TOTAL} runs finished."
echo " Summary: ${SUMMARY}"
echo "============================================================"