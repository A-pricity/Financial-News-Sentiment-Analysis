#!/usr/bin/env python
"""K-Fold Cross-Validation Dataset Preparation Script"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from pathlib import Path


def prepare_kfold_dataset(
    dataset_name: str,
    dataset_dir: str = "datasets",
    k: int = 10,
    random_seed: int = 42,
):
    """Generate K-fold splits from source dataset."""

    project_root = Path(__file__).parent.parent
    source_path = os.path.join(project_root, dataset_dir, dataset_name)
    output_dir = os.path.join(project_root, dataset_dir, f"{dataset_name}_kfold")

    train_file = os.path.join(source_path, "train.csv")
    val_file = os.path.join(source_path, "val.csv")
    test_file = os.path.join(source_path, "test.csv")
    data_file = os.path.join(source_path, "data.csv")

    if (
        os.path.exists(train_file)
        and os.path.exists(val_file)
        and os.path.exists(test_file)
    ):
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        print(
            f"Loaded pre-split files: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
    elif os.path.exists(data_file):
        df = pd.read_csv(data_file)
        train_ratio = 0.8
        val_ratio = 0.1
        total = train_ratio + val_ratio + 0.1
        train_df, temp = train_test_split(
            df, train_size=train_ratio / total, random_state=random_seed
        )
        val_df, test_df = train_test_split(
            temp, train_size=val_ratio / (val_ratio + 0.1), random_state=random_seed
        )
        print(
            f"Auto-split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
    else:
        raise FileNotFoundError(f"No data found in {source_path}")

    os.makedirs(output_dir, exist_ok=True)

    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Saved test.csv: {len(test_df)} samples")

    full_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Total samples for K-fold: {len(full_df)}")

    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_df), 1):
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_fold_df = full_df.iloc[train_idx]
        val_fold_df = full_df.iloc[val_idx]

        train_fold_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_fold_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)

        train_idx_df = pd.DataFrame({"index": train_idx})
        val_idx_df = pd.DataFrame({"index": val_idx})
        train_idx_df.to_csv(os.path.join(fold_dir, "train_idx.csv"), index=False)
        val_idx_df.to_csv(os.path.join(fold_dir, "val_idx.csv"), index=False)

        print(f"Fold {fold}: train={len(train_fold_df)}, val={len(val_fold_df)}")

    print(f"\nK-fold dataset saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare K-fold cross-validation dataset"
    )
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument("--k", type=int, default=10, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config_path = os.path.join(project_root, "configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = (
        args.dataset
        if args.dataset
        else config["data"].get("dataset_name", "financial_sentiment")
    )
    dataset_dir = config["data"].get("dataset_dir", "datasets")

    prepare_kfold_dataset(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        k=args.k,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
