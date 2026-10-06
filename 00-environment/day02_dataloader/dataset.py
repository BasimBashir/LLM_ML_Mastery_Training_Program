from itertools import product
from time import perf_counter

import torch
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset


class AGNewsDataset(Dataset):
    """
    PyTorch Dataset wrapper around Hugging Face AG News.

    Each item:
        text  -> str
        label -> int
    """

    def __init__(self, split="train", transform=None):
        self.transform = transform

        # Download/load AG News.
        self.data = hf_load_dataset(
            "fancyzhx/ag_news",
            split=split,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item["text"]
        label = item["label"]

        if self.transform:
            text = self.transform(text)

        return text, label


def custom_collate_fn(batch):
    """
    Convert a list of (text, label) samples into
    two Python lists.
    """

    texts, labels = zip(*batch)

    return list(texts), list(labels)


def load_dataset(
    split="train",
    transform=None,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=None,
):
    dataset = AGNewsDataset(
        split=split,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def batch_inspection(dataloader, num_batches=3):

    for i, (texts, labels) in enumerate(dataloader):

        print(f"Batch {i + 1}:")
        print(f"Number of samples: {len(texts)}")

        print("\nFirst 2 texts:")
        for text in texts[:2]:
            print(f"  {text}")

        print(f"\nLabels: {labels}")

        print("-" * 70)

        if i + 1 >= num_batches:
            break


def benchmark_dataloader(
    split="train",
    batch_sizes=(1, 8, 32),
    num_workers_list=(0, 1, 2, 4),
    shuffle_options=(False, True),
):

    print(
        f"{'batch_size':>10} | "
        f"{'workers':>7} | "
        f"{'shuffle':>7} | "
        f"{'samples/sec':>15} | "
        f"{'batches/sec':>15}"
    )

    print("-" * 75)

    for batch_size, num_workers, shuffle in product(
        batch_sizes,
        num_workers_list,
        shuffle_options,
    ):

        dataloader = load_dataset(
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        start = perf_counter()

        samples = 0
        batches = 0

        for texts, labels in dataloader:
            samples += len(texts)
            batches += 1

        elapsed = perf_counter() - start

        samples_per_sec = samples / elapsed
        batches_per_sec = batches / elapsed

        print(
            f"{batch_size:10} | "
            f"{num_workers:7} | "
            f"{str(shuffle):>7} | "
            f"{samples_per_sec:15.2f} | "
            f"{batches_per_sec:15.2f}"
        )


if __name__ == "__main__":

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    dataloader = load_dataset(
        split="train",
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    print(f"Dataset size: {len(dataloader.dataset)}")
    print()

    # --------------------------------------------------
    # 2. Inspect batches
    # --------------------------------------------------

    batch_inspection(
        dataloader,
        num_batches=2,
    )

    # --------------------------------------------------
    # 3. Benchmark
    # --------------------------------------------------

    benchmark_dataloader(
        split="train",
        batch_sizes=(1, 8, 32),
        num_workers_list=(0, 1, 2, 4),
        shuffle_options=(False, True),
    )