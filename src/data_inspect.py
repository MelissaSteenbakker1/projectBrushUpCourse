import os
from pathlib import Path
import numpy as np
from importdata import load_chestmnist_dataset

def summarize_split(name, dataset):
    imgs = dataset.imgs
    labels = dataset.labels

    print(f"\n{name.upper()} SPLIT")
    print(f"Number of samples: {len(dataset)}")
    print(f"Image array shape: {imgs.shape}")
    print(f"Label array shape: {labels.shape}  (N, number of labels)")
    print(f"Image dtype / range: {imgs.dtype}, min={imgs.min()}, max={imgs.max()}")

def check_labels(name, dataset):
    # Inspect the labels of the train dataset
    labels = dataset.labels
    print(f"\n{name.upper()} SPLIT")

    # Check if labels are binary
    print("Unique label values:", np.unique(labels))

    # Amount of positives per label
    label_sums = labels.sum(axis=0)
    print("Positives per label (train dataset):", label_sums)

    # Amount of samples with at least one positive label
    any_pos = (labels.sum(axis=1) > 0)
    print("Number of samples with any positive label:", any_pos.sum())

    # Amount of samples with all zeroes
    all_zero = (labels.sum(axis=1) == 0)
    print("Number of all-zero samples:", all_zero.sum())

    # Show a few examples that actually have positive labels
    pos_indices = np.where(any_pos)[0][:5]
    for idx in pos_indices:
        print(f"Sample {idx} labels:", labels[idx])

# Load dataset splits
train_dataset = load_chestmnist_dataset("train")
val_dataset = load_chestmnist_dataset("val")
test_dataset = load_chestmnist_dataset("test")

# Print basic info
summarize_split("train", train_dataset)
summarize_split("val", val_dataset)
summarize_split("test", test_dataset)
# Print label info
check_labels("train", train_dataset)
check_labels("val", val_dataset)
check_labels("test", test_dataset)

train_dataset.montage(length=20, save_folder="outputs/images/")
# Image can be viewed in the outputs/images folder



