#!/usr/bin/env bash
set -euo pipefail

# RUN_4D_US=1 RUN_5D_US=1 ./run_gd_phantom_cones_with_motion.sh

REPO_DIR=${REPO_DIR:-/gpfs/projects/KeeGroup/lzhou/5dLiverMRI-MedIA}
INPUT_DIR=${INPUT_DIR:-/gpfs/projects/KeeGroup/lzhou/datasets/Gd_Phantom/Gd_Phantom_Cones_With_Motion}
OUTPUT_DIR=${OUTPUT_DIR:-/gpfs/projects/KeeGroup/lzhou/5dLiverMRI-MedIA/output/Gd_Phantom/Gd_Phantom_Cones_With_Motion/after}
CONDA_SH=${CONDA_SH:-/gpfs/projects/KeeGroup/lzhou/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-cones-init}
CFL_DIR=${CFL_DIR:-/gpfs/projects/KeeGroup/lzhou/Cones/recon/bin}
NUM_GPUS=${NUM_GPUS:-4}
NUM_BINS=${NUM_BINS:-4}
MAX_ITER=${MAX_ITER:-300}
DEVICE=${DEVICE:-1}
UNDERSAMPLING=${UNDERSAMPLING:-10}
MPIEXEC=${MPIEXEC:-mpiexec}
MPIEXEC_LAUNCHER=${MPIEXEC_LAUNCHER:-fork}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}

RUN_RESP=${RUN_RESP:-0}
RUN_MPS=${RUN_MPS:-0}
RUN_GRIDDING=${RUN_GRIDDING:-0}
RUN_4D=${RUN_4D:-1}
RUN_5D=${RUN_5D:-0}
RUN_4D_US=${RUN_4D_US:-0}
RUN_5D_US=${RUN_5D_US:-0}

export PYTHONPATH="${CFL_DIR}:${PYTHONPATH:-}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

mkdir -p "${OUTPUT_DIR}/logs"
cd "${REPO_DIR}"

run_step() {
    local name="$1"
    shift
    local rc=0

    echo "[$(date '+%F %T')] START ${name}"
    {
        printf 'COMMAND:'
        printf ' %q' "$@"
        printf '\n\n'
    } > "${OUTPUT_DIR}/logs/${name}.log"

    set +e
    /usr/bin/time -p -o "${OUTPUT_DIR}/logs/${name}.time.txt" "$@" \
        >> "${OUTPUT_DIR}/logs/${name}.log" 2>&1
    rc=$?
    set -e

    echo "${rc}" > "${OUTPUT_DIR}/logs/${name}.exit_code.txt"
    if [[ "${rc}" == "0" ]]; then
        echo "[$(date '+%F %T')] DONE  ${name}"
    else
        echo "[$(date '+%F %T')] FAIL  ${name} rc=${rc}"
        if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
            exit "${rc}"
        fi
    fi
}

run_recon_step() {
    local name="$1"
    local script="$2"
    local output_name="$3"
    shift 3

    if (( NUM_GPUS > 1 )); then
        cmd=("${MPIEXEC}" -launcher "${MPIEXEC_LAUNCHER}" -n "${NUM_GPUS}" python "${script}" "$@"
            --multi_gpu --show_pbar --verbose
            "${INPUT_DIR}" "${OUTPUT_DIR}/${output_name}")
    else
        cmd=(python "${script}" "$@"
            --device "${DEVICE}" --show_pbar --verbose
            "${INPUT_DIR}" "${OUTPUT_DIR}/${output_name}")
    fi

    run_step "${name}" "${cmd[@]}"
}

if [[ "${RUN_RESP}" == "1" ]]; then
    cmd=(python recon_respiratory_signal.py --verbose "${INPUT_DIR}" resp)
    run_step 01_resp "${cmd[@]}"
fi

if [[ "${RUN_MPS}" == "1" ]]; then
    cmd=(python recon_coil_sensitivity.py --device "${DEVICE}" --show_pbar --verbose "${INPUT_DIR}" mps)
    run_step 02_mps "${cmd[@]}"
fi

if [[ "${RUN_GRIDDING}" == "1" ]]; then
    cmd=(python recon_gridding_motion_averaged.py --device "${DEVICE}" --verbose "${INPUT_DIR}" "${OUTPUT_DIR}/gridding_motion_averaged")
    run_step 03_gridding "${cmd[@]}"
fi

if [[ "${RUN_4D}" == "1" ]]; then
    run_recon_step 04_recon4d recon_4D_motion_resolved_PDHG.py recon_4D \
        --num_bins "${NUM_BINS}" --lambda1 1e-6 --max_iter "${MAX_ITER}"
fi

if [[ "${RUN_5D}" == "1" ]]; then
    run_recon_step 05_recon5d recon_5D_motion_resolved_PDHG.py recon_5D \
        --num_bins "${NUM_BINS}" --lambda1 1e-6 --lambda2 6e-7 --lambda3 1e-6 \
        --max_iter "${MAX_ITER}"
fi

if [[ "${RUN_4D_US}" == "1" ]]; then
    run_recon_step 06_recon4d_undersampling recon_4D_motion_resolved_PDHG_undersampling.py recon_4D_undersampling \
        --num_bins "${NUM_BINS}" --lambda1 1e-6 --undersampling "${UNDERSAMPLING}" \
        --max_iter "${MAX_ITER}"
fi

if [[ "${RUN_5D_US}" == "1" ]]; then
    run_recon_step 07_recon5d_undersampling recon_5D_motion_resolved_PDHG_undersampling.py recon_5D_undersampling \
        --num_bins "${NUM_BINS}" --lambda1 1e-6 --lambda2 6e-7 --lambda3 1e-6 \
        --undersampling "${UNDERSAMPLING}" --max_iter "${MAX_ITER}"
fi

echo "Output: ${OUTPUT_DIR}"
echo "Logs:   ${OUTPUT_DIR}/logs"
