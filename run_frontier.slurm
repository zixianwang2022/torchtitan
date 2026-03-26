#!/bin/bash

#SBATCH -A gen150
#SBATCH -J torchtitan
#SBATCH -o logs/titan-%j.o
#SBATCH -e logs/titan-%j.e
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N {{NODES}}
#SBATCH -d singleton
#SBATCH -q debug



# --- 2. Templatized Variable Definitions ---
BATCH_SIZE={{BATCH_SIZE}}
NUM_BATCHES={{NUM_BATCHES}}
PP_SIZE={{PP_SIZE}}
EP_PARALLEL_SIZE={{EP_PARALLEL_SIZE}}
MP_SIZE={{MP_SIZE}}
NODES={{NODES}}
TOTAL_GPUS={{TOTAL_GPUS}}
TRAIN_ITERS={{TRAIN_ITERS}}
MODEL_SIZE="{{MODEL_SIZE}}"





export TITAN_HOME=/lustre/orion/gen150/scratch/zixianw4/torchtitan
export PATH=$TITAN_HOME/bin:$PATH


# --- 2. Diagnostic Check (Check Logs if this fails) ---
echo "------------------------------------------------------"
echo "DEBUG: Checking Python Environment"
echo "which python:   $(which python)"
echo "python version: $(python --version)"
python -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'ROCm/CUDA: {torch.cuda.is_available()}')"
echo "------------------------------------------------------"


# --- 3. New System Network Configuration (Slingshot/HPE) ---
# These settings are crucial for the new system (Frontier/Crusher style nodes)
export HIGH_SPEED_INTERFACE="hsn0"
export NCCL_SOCKET_IFNAME=${HIGH_SPEED_INTERFACE}
export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=0

# Libfabric / Slingshot Tuning
export FI_MR_CACHE_MONITOR=kdreg2     # Critical for memory registration
export FI_CXI_DEFAULT_CQ_SIZE=262144  # Increased completion queue size
export FI_CXI_DEFAULT_TX_SIZE=8196    # Increased transmit size
export FI_CXI_RX_MATCH_MODE=hybrid    # Software fallback if needed

# General Torch Settings
export OMP_NUM_THREADS=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# --- 4. Master Node Discovery (New System Logic) ---
export MASTER_PORT=29500

# Get the list of nodes from Slurm
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}

# SSH into head node to get the IP specifically for 'hsn0'
# Replaced old 10.0.200.x logic with the new hsn0 logic
export MASTER_ADDR=$(ssh $head_node "ip -4 addr show ${HIGH_SPEED_INTERFACE} | grep 'inet' | awk '{print \$2}' | cut -d/ -f1")

if [[ -z "${MASTER_ADDR}" ]]; then
    echo "FATAL: Could not determine MASTER_ADDR for interface '${HIGH_SPEED_INTERFACE}' on master node '${head_node}'."
    exit 1
fi

echo "------------------------------------------------------"
echo "Head Node: $head_node"
echo "Head Node IP (MASTER_ADDR): $MASTER_ADDR"
echo "Interface: $HIGH_SPEED_INTERFACE"
echo "------------------------------------------------------"

# --- 4. FIX: Internet & Proxy Settings ---
# FIX: 'httpx' library requires 'socks5://', 'socks://' will cause a ValueError crash
export all_proxy=socks5://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/

# FIX: Added *.frontier.olcf.ornl.gov to prevent internal traffic from hitting the proxy (Fixes errno 97)
export no_proxy="localhost,127.0.0.0/8,*.ccs.ornl.gov,*.ncrc.gov,.olcf.ornl.gov,*.frontier.olcf.ornl.gov,${head_node}"


# --- 5. FIX: Hugging Face Timeouts ---
export HF_HOME=/lustre/orion/gen150/scratch/zixianw4/huggingface_cache
# Increase timeout to 10 minutes to handle the slow proxy
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HTTP_TIMEOUT=300

# --- 4. Execution ---
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_8b.toml"}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml"}




# 1. Set the cache to the same place your job uses
export HF_HOME=/lustre/orion/gen150/scratch/zixianw4/huggingface_cache

# 2. Run the warm-up
python3 -c "
import os
from datasets import load_dataset

# Optional: If you have a token, it helps avoid rate limits
# from huggingface_hub import login
# login(token='hf_...') 

print('--- Warming up C4 Metadata ---')
# This fetches the README, file list, and split info
ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)

print('--- Fetching one sample to ensure connectivity ---')
# This forces the network to actually establish the data stream connection
print(next(iter(ds))['text'][:50])
print('--- Success! Metadata is cached. ---')
"



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

# # dcgmi profile --resume

export TRITON_DISABLE_CACHE=1

echo "Starting TorchTitan via srun..."
echo "Master Address: $MASTER_ADDR:$MASTER_PORT"


export LAUNCHER="
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID
export WORLD_SIZE=\$SLURM_NTASKS
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=1

echo \"Rank \$RANK (Local \$LOCAL_RANK) on \$(hostname)\"

exec python -m torchtitan.train --job.config_file ${CONFIG_FILE} $@
"

# --- RUN SRUN ---
echo "LOCAL_RANK=$SLURM_LOCALID"

srun -N $SLURM_NNODES -n 16\
     --ntasks-per-node=8 \
     --gpus-per-node=8 \
     -c2 \
     --export=ALL \
     bash -c "$LAUNCHER"

