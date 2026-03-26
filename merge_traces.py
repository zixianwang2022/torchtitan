import os
import json

# --- Configuration ---
# Path to the directory containing all the rank trace files
TRACE_DIR = "/lustre/orion/gen150/scratch/zixianw4/torchtitan/outputs/titan_63B_titan_N12_PP12_EP8_MBS1_NBS50/profile_trace/iteration_4/"

# Name for the final combined trace file
OUTPUT_MERGED_TRACE = "merged_trace_stages_0-2_11.json"

# Define your parallelism dimensions
PIPELINE_STAGES = 12
TENSOR_PARALLEL_SIZE = 8

# Define which pipeline stages you want to include in the merged trace
STAGES_TO_INCLUDE = [0, 1, 2, 11]
# --- End Configuration ---

def get_ranks_for_stages(stages, pp_size, tp_size):
    """Calculates the global rank numbers for the given pipeline stages."""
    ranks_to_include = []
    for stage_idx in stages:
        if stage_idx >= pp_size:
            print(f"Warning: Stage {stage_idx} is out of bounds for {pp_size} pipeline stages. Skipping.")
            continue
        start_rank = stage_idx * tp_size
        end_rank = start_rank + tp_size
        ranks_to_include.extend(range(start_rank, end_rank))
    return sorted(ranks_to_include)

def merge_traces_manually():
    """Finds, loads, and merges specific trace files manually for compatibility."""
    ranks_to_merge = get_ranks_for_stages(STAGES_TO_INCLUDE, PIPELINE_STAGES, TENSOR_PARALLEL_SIZE)
    print(f"Will merge traces for the following {len(ranks_to_merge)} ranks:\n{ranks_to_merge}\n")

    files_to_merge = []
    for rank in ranks_to_merge:
        trace_file = os.path.join(TRACE_DIR, f"rank{rank}_trace.json")
        if os.path.exists(trace_file):
            files_to_merge.append(trace_file)
        else:
            print(f"Warning: Could not find trace file for rank {rank}: {trace_file}")

    if not files_to_merge:
        print("Error: No trace files found to merge. Please check TRACE_DIR.")
        return

    print(f"Found {len(files_to_merge)} trace files. Manually merging into {OUTPUT_MERGED_TRACE}...")
    
    all_trace_events = []
    for trace_file in files_to_merge:
        with open(trace_file, "r") as f:
            trace_data = json.load(f)
            # Each trace file has a 'traceEvents' key containing a list of events
            if 'traceEvents' in trace_data:
                all_trace_events.extend(trace_data['traceEvents'])

    # Wrap the combined list of events in the standard Chrome Trace format
    merged_json = {"traceEvents": all_trace_events}

    with open(OUTPUT_MERGED_TRACE, "w") as f:
        json.dump(merged_json, f)
        
    print(f"Successfully created merged trace file: {OUTPUT_MERGED_TRACE}")

if __name__ == "__main__":
    merge_traces_manually()