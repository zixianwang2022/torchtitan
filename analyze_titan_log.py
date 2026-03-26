import re
import sys
import argparse
import os
import pandas as pd
from datetime import datetime
import statistics

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze TorchTitan log files and store results in XLSX.")
    
    # --- File Input ---
    parser.add_argument("log_file", type=str, help="Path to the log file.")
    
    # --- Job Metadata ---
    parser.add_argument("--slurm_job_id", type=str, required=True, help="SLURM Job ID.")
    parser.add_argument("--model_size", type=str, default="16B", required=False, help="Size of the model (e.g., 16B, 63B).")
    parser.add_argument("--moe_type", type=str, default="TorchTitan", required=False, help="Type of the MoE implementation.")

    # --- Parallelism Configs ---
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus_per_node", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True, help="Pipeline Parallel size.")
    parser.add_argument("--ep", type=int, required=True, help="Expert Parallel size.")
    parser.add_argument("--dp", type=int, required=True, help="Data Parallel size.")
    parser.add_argument("--tp", type=int, required=True, help="Tensor Parallel size.")
    
    # --- Training Configs ---
    parser.add_argument("--gbs", type=int, required=True, help="Global Batch Size.")
    parser.add_argument("--mbs", type=int, required=True, help="Micro Batch Size.")
    parser.add_argument("--iterations", type=int, required=True, help="Total expected iterations.")
    parser.add_argument("--warmup_steps", type=int, default=5, required=False, help="Steps to skip for avg calculation.")
    
    # --- Model Architecture ---
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--num_experts", type=int, required=True)
    parser.add_argument("--expert_dim", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    
    # --- Optimization ---
    parser.add_argument("--activation_checkpointing", type=str, required=True)
    parser.add_argument("--pp_schedule", type=str, required=True)

    return parser.parse_args()

def analyze_log(log_file, warmup_steps):
    """
    Parses TorchTitan logs.
    Format example:
    [titan] ... rank=31  step:  5  loss: 13.9187  grad_norm: 42.5709  memory: 61.54GiB(96.17%)  tps: 386  tflops: 17.63  mfu: 9.21
    """
    # Allowed commas in regex numerical captures to prevent mismatching large values like tps: 2,630
    log_pattern = re.compile(
        r"rank=\d+\s+"
        r"step:\s+(\d+)\s+"
        r"loss:\s+([0-9.,]+)\s+"
        r"grad_norm:\s+([0-9.,]+)\s+"
        r"memory:\s+([0-9.,]+)GiB\([0-9.,]+%\)\s+"
        r"tps:\s+([0-9.,]+)\s+"
        r"tflops:\s+([0-9.,]+)\s+"
        r"mfu:\s+([0-9.,]+)"
    )

    data = { "steps": [], "loss": [], "memory":[], "tps": [], "tflops": [], "mfu":[] }

    with open(log_file, "r") as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                # Removed commas before casting to int/float
                data["steps"].append(int(match.group(1).replace(',', '')))
                data["loss"].append(float(match.group(2).replace(',', '')))
                data["memory"].append(float(match.group(4).replace(',', '')))
                data["tps"].append(float(match.group(5).replace(',', '')))
                data["tflops"].append(float(match.group(6).replace(',', '')))
                data["mfu"].append(float(match.group(7).replace(',', '')))

    if not data["steps"]: return None

    # Filter out warmup steps for averaging
    start_idx = warmup_steps if len(data["steps"]) > warmup_steps else 0
        
    tflops_for_calc = data["tflops"][start_idx:]
    
    avg_stats = {
        "matched_iterations": len(data["steps"]),
        "final_loss": data["loss"][-1] if data["loss"] else 0.0,
        "avg_tps": statistics.mean(data["tps"][start_idx:]) if data["tps"][start_idx:] else 0.0,
        "avg_tflops": statistics.mean(tflops_for_calc) if tflops_for_calc else 0.0,
        "tflops_std_dev": statistics.stdev(tflops_for_calc) if len(tflops_for_calc) > 1 else 0.0,
        "avg_mfu": statistics.mean(data["mfu"][start_idx:]) if data["mfu"][start_idx:] else 0.0,
        "peak_mem_gb": max(data["memory"]) if data["memory"] else 0.0,
        "loss_history": data["loss"]
    }
    
    return avg_stats

def write_to_xlsx(stats, args):
    """Writes the collected data to a single XLSX file."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}-titan-results.xlsx"
    
    run_id = f"{args.moe_type}-{args.model_size}-N{args.nodes}-PP{args.pp}-EP{args.ep}-GBS{args.gbs}-{args.slurm_job_id}"

    header =[
        "Name", "MOE Type", "Model Size", "AVG TFLOPs", "TFLOPs Std Dev", 
        "AVG MFU (%)", "AVG TPS (Tokens/Sec)",
        "Activation Checkpointing", "PP Schedule",
        "Iterations", "Actual Iterations",
        "Nodes", "GPUs/Node", "PP", "EP", "DP", "TP", "GBS", "MBS",
        "Final Loss", "Peak Mem (GiB)",
        "SeqLen", "Num Layers", "Hidden Dim", "Num Experts", "Top-K",
        "SLURM_JOB_ID"
    ]

    row = {
        "Name": run_id, "MOE Type": args.moe_type, "Model Size": args.model_size,
        "AVG TFLOPs": f"{stats['avg_tflops']:.2f}", 
        "TFLOPs Std Dev": f"{stats['tflops_std_dev']:.2f}",
        "AVG MFU (%)": f"{stats['avg_mfu']:.2f}",
        "AVG TPS (Tokens/Sec)": f"{stats['avg_tps']:.2f}",
        "Activation Checkpointing": args.activation_checkpointing,
        "PP Schedule": args.pp_schedule,
        "Iterations": args.iterations, 
        "Actual Iterations": stats['matched_iterations'],
        "Nodes": args.nodes, "GPUs/Node": args.gpus_per_node,
        "PP": args.pp, "EP": args.ep, "DP": args.dp, "TP": args.tp,
        "GBS": args.gbs, "MBS": args.mbs,
        "Final Loss": f"{stats['final_loss']:.4f}",
        "Peak Mem (GiB)": f"{stats['peak_mem_gb']:.2f}",
        "SeqLen": args.seqlen,
        "Num Layers": args.num_layers,
        "Hidden Dim": args.hidden_dim,
        "Num Experts": args.num_experts,
        "Top-K": args.topk,
        "SLURM_JOB_ID": args.slurm_job_id
    }

    df_main = pd.DataFrame([row])[header]
    
    # Loss Sheet
    loss_data = {"id": run_id}
    for i, l in enumerate(stats["loss_history"]):
        loss_data[f"step_{i+1}"] = l
    df_loss = pd.DataFrame([loss_data])

    try:
        with pd.ExcelFile(filename) as xls:
            old_main = pd.read_excel(xls, 'Main_Results')
            old_loss = pd.read_excel(xls, 'Loss_per_Iteration')
        new_main = pd.concat([old_main, df_main], ignore_index=True)
        new_loss = pd.concat([old_loss, df_loss], ignore_index=True)
    except FileNotFoundError:
        new_main = df_main
        new_loss = df_loss

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        new_main.to_excel(writer, sheet_name='Main_Results', index=False)
        new_loss.to_excel(writer, sheet_name='Loss_per_Iteration', index=False)
        
    return filename

def main():
    args = parse_arguments()
    stats = analyze_log(args.log_file, args.warmup_steps)

    if stats:
        outfile = write_to_xlsx(stats, args)

        metric_names =["Log File", "Output Excel", "Expected Iterations", "Actual Iterations",
                        "Average TFLOPs", "TFLOPs Std Dev", "Average MFU", "Average Tokens/Sec",
                        "Peak Memory (GiB)", "Final Loss"]
        max_width = max(len(name) for name in metric_names)
        align_width = max_width + 2

        print(f'\n================================================================================\n'
              f'--- TorchTitan Performance Analysis ---\n'
              f'================================================================================')

        print(f"{'Log File:':<{align_width}} {args.log_file}")
        print(f"{'Output Excel:':<{align_width}} {outfile}")
        print("-" * 80)
        print(f"{'Expected Iterations:':<{align_width}} {args.iterations}")
        print(f"{'Actual Iterations:':<{align_width}} {stats['matched_iterations']}")
        print(f"{'Average TFLOPs:':<{align_width}} {stats['avg_tflops']:.2f}")
        print(f"{'TFLOPs Std Dev:':<{align_width}} {stats['tflops_std_dev']:.2f}")
        print(f"{'Average MFU:':<{align_width}} {stats['avg_mfu']:.2f} %")
        print(f"{'Average Tokens/Sec:':<{align_width}} {stats['avg_tps']:.2f}")
        print(f"{'Peak Memory (GiB):':<{align_width}} {stats['peak_mem_gb']:.2f}")
        print(f"{'Final Loss:':<{align_width}} {stats['final_loss']:.4f}")
        print(f'================================================================================\n')

    else:
        print("No valid training steps found in log file.")

if __name__ == "__main__":
    main()