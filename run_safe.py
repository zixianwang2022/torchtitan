# # run_safe.py
# import os
# import sys
# import runpy

# def main():
#     # 1. Get the Local Rank (automatically set by torchrun)
#     local_rank = os.environ.get("LOCAL_RANK", "0")
#     job_id = os.environ.get("SLURM_JOB_ID", "unknown_job")

#     # 2. Define a unique cache directory for this process
#     # Using /tmp ensures it is fast and local to the node
#     unique_cache_path = f"/tmp/titan_triton_cache_{job_id}/rank_{local_rank}"
    
#     # 3. Create the directory
#     os.makedirs(unique_cache_path, exist_ok=True)

#     # 4. Set the Environment Variable BEFORE importing Triton/TorchTitan
#     os.environ["TRITON_CACHE_DIR"] = unique_cache_path
    
#     print(f"[SafeWrapper] Rank {local_rank} -> TRITON_CACHE_DIR={unique_cache_path}")

#     # 5. Execute the actual TorchTitan training module
#     # This is equivalent to running "python -m torchtitan.train"
#     sys.argv[0] = "torchtitan.train"
#     runpy.run_module("torchtitan.train", run_name="__main__", alter_sys=True)

# if __name__ == "__main__":
#     main()
# run_safe.py
import os
import sys
import runpy

def main():
    local_rank = os.environ.get("LOCAL_RANK", "0")
    job_id = os.environ.get("SLURM_JOB_ID", "unknown_job")

    # 1. Unique Cache Directory (Keep this!)
    unique_cache_path = f"/tmp/titan_triton_cache_{job_id}/rank_{local_rank}"
    os.makedirs(unique_cache_path, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = unique_cache_path
    
    # 2. FIX: Enable Subprocess Autotuning
    # This prevents "Illegal Memory Access" in one kernel from killing the whole job.
    os.environ["TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC"] = "1"

    print(f"[SafeWrapper] Rank {local_rank} -> Cache: {unique_cache_path} | Autotune Subproc: ON")

    # 3. Run Training
    sys.argv[0] = "torchtitan.train"
    runpy.run_module("torchtitan.train", run_name="__main__", alter_sys=True)

if __name__ == "__main__":
    main()