import os
import pickle

# --- Configuration ---
SNAPSHOT_DIR = "/lustre/orion/gen150/scratch/zixianw4/torchtitan/outputs/titan_63B_titan_N12_PP12_EP8_MBS1_NBS50/memory_snapshot/iteration_4/"
PIPELINE_STAGES = 12
TENSOR_PARALLEL_SIZE = 8
STAGES_TO_ANALYZE = [0, 1, 2, 11]
TOP_N_ALLOCATIONS = 10  # Increased to show more detail
# --- End Configuration ---

def format_size(size_bytes):
    """Formats bytes into KB, MB, GB, etc."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = 0
    while size_bytes >= 1024 and i < len(size_name) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f}{size_name[i]}"

def analyze_snapshot_file(filepath):
    """
    Loads a single pickle file and prints a summary of its memory usage.
    This version is tailored to the data structure found in the ROCm/torchtitan snapshot.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"\n--- Analyzing {os.path.basename(filepath)} ---")
    with open(filepath, "rb") as f:
        snapshot = pickle.load(f)

    if 'segments' not in snapshot:
        print("Error: Snapshot dictionary does not contain a 'segments' key.")
        return

    total_allocated = 0
    active_allocations = []
    
    # Iterate through each large memory segment
    for segment in snapshot['segments']:
        # Then iterate through the individual blocks within that segment
        for block in segment['blocks']:
            # The key is 'state' and the value for an in-use block is 'active'
            if block.get('state') == 'active':
                total_allocated += block['size']
                active_allocations.append({
                    'size': block['size'],
                    # The traceback key is 'frames'
                    'traceback': block.get('frames', [])
                })

    print(f"Total Active Memory: {format_size(total_allocated)}")

    # Sort allocations by size to find the largest ones
    active_allocations.sort(key=lambda x: x['size'], reverse=True)

    print(f"\nTop {TOP_N_ALLOCATIONS} Largest Active Allocations:")
    for i, alloc in enumerate(active_allocations[:TOP_N_ALLOCATIONS]):
        print(f"  {i+1}. Size: {format_size(alloc['size'])}")
        
        # The traceback is stored in the 'traceback' key we created
        if alloc['traceback']:
            # Show the most recent (most specific) calls from the stack
            for frame in alloc['traceback'][-4:]: 
                filename, lineno, func, line = frame
                # Make the path shorter for readability
                short_filename = os.path.join(os.path.basename(os.path.dirname(filename)), os.path.basename(filename))
                print(f"     -> in .../{short_filename}:{lineno} (in {func})")
                print(f"        Code: '{line.strip()}'")
        else:
            print("     -> (Traceback not available for this allocation)")


def get_ranks_for_stages(stages, pp_size, tp_size):
    """Helper function to map stages to ranks."""
    ranks = {}
    for stage_idx in stages:
        if stage_idx >= pp_size:
            continue
        start_rank = stage_idx * tp_size
        end_rank = start_rank + tp_size
        ranks[stage_idx] = list(range(start_rank, end_rank))
    return ranks

def main():
    stage_to_ranks = get_ranks_for_stages(STAGES_TO_ANALYZE, PIPELINE_STAGES, TENSOR_PARALLEL_SIZE)

    for stage, ranks in stage_to_ranks.items():
        print(f"\n{'='*25} Analyzing Pipeline Stage {stage} {'='*25}")
        # Within a Tensor Parallel group, memory usage should be nearly identical.
        # Analyzing the first rank of the stage gives a representative sample.
        rank_to_analyze = ranks[0]
        snapshot_file = os.path.join(SNAPSHOT_DIR, f"rank{rank_to_analyze}_memory_snapshot.pickle")
        analyze_snapshot_file(snapshot_file)
        print(f"{'='*60}")

if __name__ == "__main__":
    main()