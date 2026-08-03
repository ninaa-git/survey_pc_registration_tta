#!/usr/bin/env bash
set -euo pipefail

# ==================== CONFIGURATION ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/slurm.sh"
LOGDIR="$SCRIPT_DIR/logs/slurm"
mkdir -p "$LOGDIR"

# ==================== COMPUTE CANADA RESOURCES ====================
ACCOUNT="def-egranger"
GRES="gpu:1"
CPUS=4
MEM="2G"
TIME="00:10:00"

# ==================== PATHS ====================
REPO_ROOT="${REPO_CREATE_COR_ROOT:-$HOME/pc-registration/silico/PARENet/experiments/my3DMatch_TTA}"
VENV="${VENV:-$HOME/venv}"

# ==================== CORRUPTIONS ====================
corruptions=(local_density_dec occlusion cutout)
severities=(5)

# ==================== FUNCTIONS ====================

submit_single_job() {
    local corruption=$1
    local severity=$2
    local JOB_NUM=$3
    
    # Create a unique job name
    local JOB_NAME="cor${corruption}_sev${severity}"
    
    echo "[SUBMIT] Job $JOB_NUM: corruption=$corruption, severity=$severity"
    
    sbatch \
        --job-name="$JOB_NAME" \
        --account="$ACCOUNT" \
        --gres="$GRES" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --output="${LOGDIR}/%x-%j.out" \
        --error="${LOGDIR}/%x-%j.err" \
        --export=ALL,SERVER=cc,REPO_ROOT="$REPO_ROOT",VENV="$VENV",COR="$corruption",SEV="$severity" \
        "$JOB_SCRIPT"
}

launch_all_combos() {
    echo "=================================================="
    echo "Launching creation of dataset of all corruptions"
    echo "=================================================="
    echo "Corruptions: ${corruptions[*]}"
    echo "Severities: ${severities[*]}"
    echo "Total combinations: $(((${#corruptions[@]} - 1) * ${#severities[@]} +1))"
    echo ""
    
    local job_num=1
    
    # Generate all combinations and submit jobs
    for corruption in "${corruptions[@]}"; do
        if [[ "$corruption" == "clean" ]]; then
            local severity="None"
            submit_single_job "$corruption" "$severity" "$job_num"
            job_num=$((job_num + 1))
            sleep 0.5  # Small delay to avoid overwhelming the scheduler
        else
            for severity in "${severities[@]}"; do
                submit_single_job "$corruption" "$severity" "$job_num"
                job_num=$((job_num + 1))
                sleep 0.5  # Small delay to avoid overwhelming the scheduler
            done
        fi
    done
    
    echo ""
    echo "=================================================="
    echo "All jobs submitted!"
    echo "=================================================="
    echo "Total jobs submitted: $((job_num - 1))"
    echo "Monitor your jobs with: squeue -u \$USER"
    echo "Cancel all jobs with: scancel -u \$USER --name=grid_*"
}

smoke_test() {
    echo "=================================================="
    echo "Running smoke test (single corruption)"
    echo "=================================================="
    
    # Use first value of each parameter
    local corruption=corruptions
    local severity="3"
    
    echo "Creation of dataset with corruption=$corruption, severity=$severity"
    
    submit_single_job "$corruption" "$severity" "smoke_test"
    
    echo "Smoke test submitted. Check logs in $LOGDIR"
    echo "Monitor your jobs with: squeue -u \$USER"
    echo "Cancel all jobs with: scancel -u \$USER --name=grid_*"
}

show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    launch              Launch full grid search
    smoke               Run a quick smoke test with first parameter combo
    help                Show this help message

Environment variables:
    REPO_ROOT          Path to repository
    VENV               Path to virtual environment

Examples:
    # Launch full corruptions evaluation
    $0 launch

    # Run smoke test
    $0 smoke

Current corruptions:
    Corruptions: ${corruptions[*]}
    Severities: ${severities[*]}
    Total combinations: $(((${#corruptions[@]} - 1) * ${#severities[@]} +1))

EOF
}

# ==================== MAIN ====================

case "${1:-}" in
    launch)
        launch_all_combos
        ;;
    
    smoke)
        smoke_test
        ;;
    
    help|--help|-h)
        show_usage
        ;;
    
    *)
        echo "ERROR: Unknown command '${1:-}'"
        echo ""
        show_usage
        exit 1
        ;;
esac