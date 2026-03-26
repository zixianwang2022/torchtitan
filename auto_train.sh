#!/bin/bash

# --- Setup Directories ---
mkdir -p gen_configs
mkdir -p gen_scripts
mkdir -p logs

TEMPLATE_TOML="torchtitan/models/deepseek_v3/train_configs/deepseek_v3_template.toml"
TEMPLATE_SLURM="frontier_titan.slurm.template"

if [ ! -f "$TEMPLATE_TOML" ] || [ ! -f "$TEMPLATE_SLURM" ]; then
    echo "ERROR: Templates not found in templates/ directory."
    exit 1
fi

echo "=========================================================="
echo "   TorchTitan Automated Experiment Submission"
echo "=========================================================="

##########################################################################
# --- 1. NODE CONFIGURATION MAPPING ---
# Key:   "NODES:GPUS_PER_NODE"
# Value: Space-separated list of "PP_DEGREE:EP_DEGREE" strategies
##########################################################################
declare -A PP_STRATEGY_MAP

# 173B_titan

# 1 Node (8 GPUs) -> PP=1 / EP=8
PP_STRATEGY_MAP["1:8"]="2:4" 
PP_STRATEGY_MAP["2:8"]="2:8"
PP_STRATEGY_MAP["4:8"]="4:8"
PP_STRATEGY_MAP["6:8"]="6:8"
PP_STRATEGY_MAP["8:8"]="8:8"
PP_STRATEGY_MAP["12:8"]="12:8"
PP_STRATEGY_MAP["16:8"]="16:8"
PP_STRATEGY_MAP["24:8"]="24:8"
PP_STRATEGY_MAP["30:8"]="30:8"
PP_STRATEGY_MAP["48:8"]="24:16"
PP_STRATEGY_MAP["60:8"]="30:16 "
PP_STRATEGY_MAP["64:8"]="8:64 "
PP_STRATEGY_MAP["96:8"]="24:32 "


##########################################################################
# --- 2. EXPERIMENT CONFIGURATIONS ---
# Key:   "NODES:GPUS_PER_NODE" (Must match keys above)
# Value: Space-separated list of experiment strings
#
# Format: MBS : NBS : TRAIN_ITERS : MODEL_SIZE : CHECKPOINT : PP_TYPE
#
# MBS         = Micro Batch Size (Batch size per GPU per forward pass)
# NBS         = Number of Micro Batches (Gradient Accumulation Steps)
# TRAIN_ITERS = Total Training Steps
# MODEL_SIZE  = Flavor (e.g., 10B_titan, 63B_titan)
# CHECKPOINT  = Activation Checkpoint Mode (selective, full, none)
# PP_TYPE     = Schedule (1F1B, Interleaved)
##########################################################################
declare -A EXP_CONFIGS

# Experiment for 1 Node (Sanity Check)
# EXP_CONFIGS["1:8"]="1:4:20:10B_titan:none:none 2:2:20:10B_titan:none:none 4:1:20:10B_titan:none:none 6:1:20:10B_titan:none:none "
# EXP_CONFIGS["2:8"]="1:1:20:10B_titan:none:none 2:1:20:10B_titan:none:none 4:1:20:10B_titan:none:none 6:1:20:10B_titan:none:none "

# Experiment for 2 Nodes
# BS=6 of all stages, NBS=2, Steps=15, Model=10B_titan, AC=["none", "selective", "full"], Schedule=1F1B
# EXP_CONFIGS["2:8"]="6:2:15:10B_titan:none:1F1B"
# EXP_CONFIGS["4:8"]="1:20:5:10B_titan:none:1F1B 1:20:5:10B_titan:full:1F1B"

# EXP_CONFIGS["1:8"]="1:1:3:63B_titan_4L:none:1F1B"

# 01/21/2026
# EXP_CONFIGS["4:8"]="1:100:8:63B_titan:full:1F1B 1:100:8:63B_titan:none:1F1B 2:100:8:63B_titan:full:1F1B 4:100:8:63B_titan:full:1F1B "
# EXP_CONFIGS["8:8"]="1:100:8:63B_titan:full:1F1B 1:100:8:63B_titan:none:1F1B 2:100:8:63B_titan:full:1F1B 4:100:8:63B_titan:full:1F1B "
# EXP_CONFIGS["16:8"]="1:100:8:63B_titan:full:1F1B 1:100:8:63B_titan:none:1F1B 2:100:8:63B_titan:full:1F1B 4:100:8:63B_titan:full:1F1B "

# debug 
# EXP_CONFIGS["2:8"]="1:8:4:63B_titan_8L:full:1F1B"
# PP_STRATEGY_MAP["8:8"]="4:8"
EXP_CONFIGS["8:8"]="  1:32:25:63B_titan:full:1F1B  "
# EXP_CONFIGS["8:8"]=" 1:16:25:63B_titan:full:1F1B  1:16:25:63B_titan:selective:1F1B  1:16:25:63B_titan:none:1F1B  "
# EXP_CONFIGS["8:8"]=" 1:32:10:63B_titan:full:1F1B  1:64:5:63B_titan:full:1F1B  1:128:5:63B_titan:full:1F1B "



# EXP_CONFIGS["8:8"]=" 1:64:5:63B_titan:full:1F1B  1:128:5:63B_titan:full:1F1B "


# EXP_CONFIGS["8:8"]=" 1:32:15:63B_titan:full:1F1B  1:512:15:63B_titan:selective:1F1B  1:512:15:63B_titan:none:1F1B "

# EXP_CONFIGS["12:8"]=" 1:24:4:173B_titan:full:1F1B  "

# PP_STRATEGY_MAP["24:8"]="12:8"
# EXP_CONFIGS["24:8"]=" 1:24:4:173B_titan:full:1F1B  "

# PP_STRATEGY_MAP["32:8"]="8:16 16:8"
# EXP_CONFIGS["32:8"]=" 1:32:4:173B_titan:full:1F1B  "


# PP_STRATEGY_MAP["30:8"]="30:8"
# EXP_CONFIGS["30:8"]=" 1:30:4:537B_titan:full:1F1B  "


# PP_STRATEGY_MAP["60:8"]="60:8"
# EXP_CONFIGS["60:8"]=" 1:60:4:537B_titan:full:1F1B  1:60:4:1T_titan:full:1F1B  "  



# PP_STRATEGY_MAP["24:8"]="12:8"
# EXP_CONFIGS["24:8"]=" 1:512:5:173B_titan:full:1F1B "
# EXP_CONFIGS["30:8"]=" 1:512:5:537B_titan:full:1F1B "
# EXP_CONFIGS["60:8"]=" 1:512:5:1T_titan:full:1F1B "



# EXP_CONFIGS["2:8"]=" 1:20:6:63B_titan_4L:none:1F1B "

# EXP_CONFIGS["8:8"]=" 1:200:8:173B_titan_4L:full:1F1B  1:200:8:537B_titan_4L:full:1F1B "

# EXP_CONFIGS["4:8"]=" 1:200:8:537B_titan_4L:full:1F1B "

# EXP_CONFIGS["4:8"]=" 1:200:7:63B_titan:full:1F1B  1:200:7:63B_titan:selective:1F1B "
# EXP_CONFIGS["6:8"]=" 1:200:7:63B_titan:full:1F1B  1:200:7:63B_titan:selective:1F1B "
# EXP_CONFIGS["8:8"]=" 1:200:7:63B_titan:full:1F1B  1:200:7:63B_titan:selective:1F1B "

# 01/21/2026
# EXP_CONFIGS["12:8"]="  1:100:7:173B_titan:full:1F1B  2:100:7:173B_titan:full:1F1B  "
# EXP_CONFIGS["24:8"]=" 1:100:7:173B_titan:full:1F1B  2:100:7:173B_titan:full:1F1B "
# EXP_CONFIGS["12:8"]=" 1:200:4:63B_titan:full:1F1B  1:200:4:63B_titan:selective:1F1B "

# EXP_CONFIGS["8:8"]=" 1:12:4:63B_titan:full:1F1B  "
# EXP_CONFIGS["12:8"]=" 1:12:4:63B_titan:full:1F1B  "
# EXP_CONFIGS["24:8"]=" 1:24:4:63B_titan:full:1F1B  "
# EXP_CONFIGS["48:8"]=" 1:48:4:63B_titan:full:1F1B  "

# EXP_CONFIGS["12:8"]="1:24:5:173B_titan:full:1F1B 1:24:5:173B_titan:selective:1F1B  " # 01/06/2026: OOM
# EXP_CONFIGS["24:8"]="1:12:5:173B_titan:full:1F1B   " # 01/06/2026: OOM
# EXP_CONFIGS["30:8"]="1:30:5:537B_titan:full:1F1B 1:30:5:537B_titan:selective:1F1B  "

# EXP_CONFIGS["4:8"]="1:10:3:63B_titan:full:1F1B "
# EXP_CONFIGS["48:8"]="1:24:3:173B_titan:full:1F1B"
# EXP_CONFIGS["64:8"]="1:24:3:173B_titan:full:1F1B"
# EXP_CONFIGS["96:8"]="1:24:3:173B_titan:full:1F1B"

# EXP_CONFIGS["4:8"]="1:100:10:63B_titan:full:1F1B 2:100:10:63B_titan:full:1F1B 3:100:10:63B_titan:full:1F1B 4:100:10:63B_titan:full:1F1B "
# EXP_CONFIGS["8:8"]="1:100:10:63B_titan:full:1F1B 2:100:10:63B_titan:full:1F1B 3:100:10:63B_titan:full:1F1B 4:100:10:63B_titan:full:1F1B "
# EXP_CONFIGS["8:8"]="2:20:10:63B_titan:full:1F1B"
# EXP_CONFIGS["2:8"]="2:10:2:10B_titan:full:1F1B"
# EXP_CONFIGS["1:8"]="2:1:6:10B_titan:none:none"
# EXP_CONFIGS["48:8"]="1:30:5:173B_titan:full:1F1B "


##########################################################################
# --- 3. SUBMISSION LOGIC ---
##########################################################################

for node_key in "${!PP_STRATEGY_MAP[@]}"; do
    IFS=':' read -r NODES GPUS_PER_NODE <<< "$node_key"
    
    # Get strategies and configs
    strategies_str=${PP_STRATEGY_MAP[$node_key]}
    configs_str=${EXP_CONFIGS[$node_key]}

    if [ -z "$configs_str" ]; then continue; fi

    for strategy in $strategies_str; do
        IFS=':' read -r PP_SIZE EP_SIZE <<< "$strategy"

        # --- Sanity Check ---
        TOTAL_AVAIL_GPUS=$(( NODES * GPUS_PER_NODE ))
        TOTAL_REQ_GPUS=$(( PP_SIZE * EP_SIZE )) 

        if (( TOTAL_AVAIL_GPUS % TOTAL_REQ_GPUS != 0 )); then
            echo "SKIPPING: Config mismatch. Avail: $TOTAL_AVAIL_GPUS not divisible by Req: $TOTAL_REQ_GPUS"
            continue
        fi

        # # Calculate Data Parallel Degree (e.g., 256 / 64 = 4)
        # DP_DEGREE=$(( TOTAL_AVAIL_GPUS / TOTAL_REQ_GPUS ))

        for config in $configs_str; do
            IFS=':' read -r MBS NBS TRAIN_ITERS MODEL_SIZE AC_MODE PP_TYPE <<< "$config"

            # # --- 1. Calculate Batch Sizes ---
            # # Local Batch Size = MBS * NBS
            # # (This is what determines how many microbatches run per step)
            # LOCAL_BATCH_SIZE=$(( MBS * NBS ))
            
            # # Global Batch Size = Local Batch Size * EP (Assuming DP=1)
            # GLOBAL_BATCH_SIZE=$(( MBS * NBS * EP_SIZE ))
            # # GLOBAL_BATCH_SIZE=$(( MBS * NBS * DP_DEGREE ))

            # --- 1. Calculate Batch Sizes ---
            LOCAL_BATCH_SIZE=$(( MBS * NBS ))

            # Calculate the actual Data Parallel degree (Total GPUs / PP_SIZE)
            DP_DEGREE=$(( TOTAL_AVAIL_GPUS / PP_SIZE ))
                        
            # Global Batch Size = Local Batch Size * DP_DEGREE
            GLOBAL_BATCH_SIZE=$(( LOCAL_BATCH_SIZE * DP_DEGREE ))


            # --- 2. Determine Sequence Length ---
            if [[ "$MODEL_SIZE" == "10B_titan" ]]; then
                SEQ_LEN=2048
            else
                SEQ_LEN=4096
            fi

            # --- 3. Map PP Schedule ---
            if [[ "$PP_TYPE" == "Interleaved" ]]; then
                PP_SCHEDULE="Interleaved1F1B"
            else
                PP_SCHEDULE="1F1B"
            fi

            # --- Job Naming ---
            JOB_NAME="titan_${MODEL_SIZE}_N${NODES}_PP${PP_SIZE}_EP${EP_SIZE}_MBS${MBS}_NBS${NBS}"
            
            GEN_TOML="gen_configs/${JOB_NAME}.toml"
            GEN_SLURM="gen_scripts/${JOB_NAME}.slurm"


            echo "-> Generating Job: $JOB_NAME"
            # echo "   [Nodes: $NODES] [PP: $PP_SIZE] [EP: $EP_SIZE] [GBS: $GLOBAL_BATCH_SIZE]"

            echo "   [MBS: $MBS] [NBS: $NBS] [LocalBS: $LOCAL_BATCH_SIZE] [SeqLen: $SEQ_LEN]"
            echo "   [Nodes: $NODES] [Model: $MODEL_SIZE] [SeqLen: $SEQ_LEN]"

            # 1. Generate TOML
            sed -e "s/{{JOB_NAME}}/${JOB_NAME}/g" \
                -e "s/{{MODEL_FLAVOR}}/${MODEL_SIZE}/g" \
                -e "s/{{MICRO_BATCH_SIZE}}/${MBS}/g" \
                -e "s/{{LOCAL_BATCH_SIZE}}/${LOCAL_BATCH_SIZE}/g" \
                -e "s/{{GLOBAL_BATCH_SIZE}}/${GLOBAL_BATCH_SIZE}/g" \
                -e "s/{{STEPS}}/${TRAIN_ITERS}/g" \
                -e "s/{{PP_DEGREE}}/${PP_SIZE}/g" \
                -e "s/{{EP_DEGREE}}/${EP_SIZE}/g" \
                -e "s/{{PP_SCHEDULE}}/${PP_SCHEDULE}/g" \
                -e "s/{{AC_MODE}}/${AC_MODE}/g" \
                -e "s/{{SEQ_LEN}}/${SEQ_LEN}/g" \
                $TEMPLATE_TOML > $GEN_TOML

            # 2. Generate SLURM
            ABS_CONFIG_PATH=$(readlink -f $GEN_TOML)
            
            sed -e "s/{{JOB_NAME}}/${JOB_NAME}/g" \
                -e "s/{{NODES}}/${NODES}/g" \
                -e "s/{{GPUS_PER_NODE}}/${GPUS_PER_NODE}/g" \
                -e "s/{{PP_SIZE}}/${PP_SIZE}/g" \
                -e "s/{{EP_SIZE}}/${EP_SIZE}/g" \
                -e "s/{{MICRO_BATCH_SIZE}}/${MBS}/g" \
                -e "s/{{LOCAL_BATCH_SIZE}}/${LOCAL_BATCH_SIZE}/g" \
                -e "s/{{GLOBAL_BATCH_SIZE}}/${GLOBAL_BATCH_SIZE}/g" \
                -e "s/{{PP_SCHEDULE}}/${PP_SCHEDULE}/g" \
                -e "s/{{NUM_BATCHES}}/${NBS}/g" \
                -e "s/{{TRAIN_ITERS}}/${TRAIN_ITERS}/g" \
                -e "s/{{MODEL_SIZE}}/${MODEL_SIZE}/g" \
                -e "s/{{AC_MODE}}/${AC_MODE}/g" \
                -e "s/{{SEQ_LEN}}/${SEQ_LEN}/g" \
                -e "s/{{TOTAL_AVAIL_GPUS}}/${TOTAL_AVAIL_GPUS}/g" \
                -e "s|{{GENERATED_CONFIG_PATH}}|${ABS_CONFIG_PATH}|g" \
                $TEMPLATE_SLURM > $GEN_SLURM

            # 3. Submit
            echo "   Submitting..."
            sbatch $GEN_SLURM
            sleep 1
        done
    done
done