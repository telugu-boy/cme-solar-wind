import argparse
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate PatchTSMixer")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (creates a folder in results/full/)")
    # Allow passing through all other args to train_patchtsmixer.py
    args, unknown = parser.parse_known_args()

    # 1. Train the model
    print(f"=== Training Model: {args.model_name} ===")
    train_cmd = [sys.executable, "-m", "experiments.train_patchtsmixer", "--model_name", args.model_name] + unknown
    subprocess.run(train_cmd, check=True)

    # The checkpoint will be saved in results/full/<model_name>/patchtsmixer_backbone_final.pt
    ckpt_path = Path("results/full") / args.model_name / "patchtsmixer_backbone_final.pt"
    
    if not ckpt_path.exists():
        print(f"Error: Expected checkpoint at {ckpt_path} but it was not found.")
        sys.exit(1)

    # 2. Evaluate the model
    print(f"\n=== Evaluating Model: {args.model_name} ===")
    eval_cmd = [sys.executable, "evaluate_model.py", "--checkpoint", str(ckpt_path)]
    subprocess.run(eval_cmd, check=True)

if __name__ == "__main__":
    main()
