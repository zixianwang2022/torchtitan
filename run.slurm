#!/bin/bash

#SBATCH -J torchtitan_multi_node
#SBATCH -o /work1/mzhang/zixianw4/torchtitan/torchtitan-%j.o
#SBATCH -e /work1/mzhang/zixianw4/torchtitan/torchtitan-%j.e
#SBATCH -t 00:10:00
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 8
#######SBATCH --exclude=k005-004,k005-005,k003-001,k004-006,k004-003

# --- 1. Robust Environment Setup ---
# Source conda.sh to get the 'conda' command
source /work1/mzhang/zixianw4/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate titan

# FIX: Force the PATH to ensure we use the 'titan' environment binaries
# even if 'conda activate' flaked out in the batch shell.
export TITAN_HOME=/work1/mzhang/zixianw4/miniconda3/envs/titan
export PATH=$TITAN_HOME/bin:$PATH


# --- 2. Diagnostic Check (Check Logs if this fails) ---
echo "------------------------------------------------------"
echo "DEBUG: Checking Python Environment"
echo "which python:   $(which python)"
echo "python version: $(python --version)"
python -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'ROCm/CUDA: {torch.cuda.is_available()}')"
echo "------------------------------------------------------"



export OMP_NUM_THREADS=2
export CUDA_DEVICE_MAX_CONNECTIONS=1


# --- 2. Network Discovery Logic (Specific to this Cluster) ---
# We use the logic from your template to find the interface on the 10.0.200.x subnet
IB_SUBNET="10.0.200."
export MASTER_PORT=29500

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

# SSH into the head node to find the interface and IP on the specific IB subnet
ib_info=$(ssh $head_node "ip -4 a | grep 'inet ${IB_SUBNET}'")

if [[ -z "${ib_info}" ]]; then
    echo "ERROR: Could not find an interface on the IB_SUBNET (${IB_SUBNET}) on master node ${head_node}."
    exit 1
fi

# Extract IP and Interface name
head_node_ip=$(echo $ib_info | awk '{print $2}' | cut -d/ -f1)
ib_interface=$(echo $ib_info | awk '{print $NF}') # Gets the last field (interface name)

echo "Head Node: $head_node"
echo "Head Node IP: $head_node_ip"
echo "Interface: $ib_interface"

# --- 3. Network Configuration ---
export NCCL_SOCKET_IFNAME=${ib_interface}
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# export NCCL_IB_DISABLE=0 
# export NCCL_P2P_DISABLE=0

# Clean up AWS/EFA/Frontier specific paths (Removed EFA LD_LIBRARY_PATH)
export CUDA_LAUNCH_BLOCKING=0
export NCCL_BUFFSIZE=2097152

# --- 4. Execution ---
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_8b.toml"}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml"}

# # =========================================================================
# # === TRITON CACHE FIX (PRE-WARM STRATEGY) ===
# # =========================================================================

# # 1. Clean the cache to remove any corrupted files from previous failed runs
# #    (Using /tmp is faster and safer than /home for caches)
# export TRITON_CACHE_DIR="/home1/zixianw4/triton_cache_${SLURM_JOB_ID}"
# mkdir -p $TRITON_CACHE_DIR
# rm -rf $TRITON_CACHE_DIR/*

# echo "Using Triton Cache: $TRITON_CACHE_DIR"

# # 2. Run a SINGLE dummy compilation.
# #    This forces Triton to build 'hip_utils.so' safely in a single process.
# #    Once built, the 8 parallel processes below will just read it (no race condition).
# echo "Pre-warming Triton cache..."
# python -c "import torch; torch.compile(lambda x: x + x)(torch.ones(1).cuda())"

# if [ $? -eq 0 ]; then
#     echo "Cache pre-warm successful."
# else
#     echo "Cache pre-warm failed! Check PyTorch/ROCm compatibility."
#     exit 1
# fi
# # =========================================================================

export TRITON_DISABLE_CACHE=1

# Note: I commented out dcgmi as it was not present in your working DeepSpeed template
# and might not be available/configured on this specific partition.
# dcgmi profile --pause

echo "Starting TorchTitan on ${head_node}..."

# Updated srun to utilize the discovered IP and 8 GPUs
# srun -N 1 -n 8 --gres=gpu:8 \
#      --rdzv_id 101 \
#      --rdzv_backend c10d \
#      --rdzv_endpoint "$head_node_ip:29500" \
#      -m torchtitan.train \
#      --job.config_file ${CONFIG_FILE} "$@"
# torchrun --nnodes 1 --nproc_per_node 8 \
#     --rdzv_id 101 \
#     --rdzv_backend c10d \
#     --rdzv_endpoint "$head_node_ip:29500" \
#     -m torchtitan.train \
#     --job.config_file ${CONFIG_FILE} "$@"

torchrun --nnodes 1 --nproc_per_node 8 \
    --rdzv_id 101 \
    --rdzv_backend c10d \
    --rdzv_endpoint "$head_node_ip:29500" \
    run_safe.py \
    --job.config_file ${CONFIG_FILE} "$@"

# dcgmi profile --resume