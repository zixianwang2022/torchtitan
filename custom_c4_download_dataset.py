# download_script.py
from datasets import load_dataset

print("Downloading C4 dataset...")
dataset = load_dataset("allenai/c4", "en")
dataset.save_to_disk("/lustre/orion/gen150/scratch/zixianw4/torchtitan/datasets/c4_en")
print("Download and save complete.")
