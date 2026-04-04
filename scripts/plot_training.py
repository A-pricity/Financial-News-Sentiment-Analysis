#!/usr/bin/env python3
"""Plot training history from CSV"""

import argparse
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_results_dir(base_dir: str = "results/training") -> str:
    """Find the latest results directory"""
    dirs = glob.glob(os.path.join(base_dir, "[0-9]*"))
    if not dirs:
        print(f"Error: No results found in {base_dir}")
        sys.exit(1)
    return max(dirs, key=os.path.getmtime)


def plot_training_history(csv_path: str, output_path: str = None):
    """Plot training and validation loss/accuracy from CSV"""

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    ax1 = axes[0]
    ax1.plot(df["epoch"], df["train_loss"], "b-", label="Train Loss", marker="o")
    ax1.plot(df["epoch"], df["val_loss"], "r-", label="Val Loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Accuracy
    ax2 = axes[1]
    ax2.plot(df["epoch"], df["train_accuracy"], "b-", label="Train Acc", marker="o")
    ax2.plot(df["epoch"], df["val_accuracy"], "r-", label="Val Acc", marker="s")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training history")
    parser.add_argument(
        "--csv",
        type=str,
        default="results/training/latest/training_history.csv",
        help="Path to training history CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/training/latest/training_history.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--latest", action="store_true", help="Use the latest results directory"
    )
    args = parser.parse_args()

    if args.latest:
        latest_dir = find_latest_results_dir()
        args.csv = os.path.join(latest_dir, "training_history.csv")
        args.output = os.path.join(latest_dir, "training_history.png")
        print(f"Using latest results: {latest_dir}")

    plot_training_history(args.csv, args.output)


if __name__ == "__main__":
    main()
