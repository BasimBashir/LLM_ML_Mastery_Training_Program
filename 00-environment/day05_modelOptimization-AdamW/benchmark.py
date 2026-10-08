import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter


def run_experiment(train_script, batch_size, accumulation_steps, args, checkpoint_dir):
    checkpoint = checkpoint_dir / f"batch_{batch_size}_accum_{accumulation_steps}.pt"
    command = [
        sys.executable,
        str(train_script),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation-steps",
        str(accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--max-features",
        str(args.max_features),
        "--seed",
        str(args.seed),
        "--checkpoint",
        str(checkpoint),
    ]
    started = perf_counter()
    print(f"\n--- batch_size={batch_size}, accumulation_steps={accumulation_steps} ---")
    subprocess.run(command, check=True)
    elapsed = perf_counter() - started

    print(f"Total runtime (training and validation): {elapsed:.2f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Compare batch size 8 with batch size 2 and four accumulated micro-batches."
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in the range [0, 1).")
    if args.max_features < 2:
        parser.error("--max-features must be at least 2.")

    train_script = Path(__file__).resolve().parents[1] / "day04_TrainingLoop" / "train.py"
    with tempfile.TemporaryDirectory(prefix="day05_batch_comparison_") as temp_dir:
        checkpoint_dir = Path(temp_dir)
        batch8_time = run_experiment(
            train_script, 8, 1, args, checkpoint_dir
        )
        batch2_time = run_experiment(
            train_script, 2, 4, args, checkpoint_dir
        )

    print("\nBoth configurations have nominal effective batch size 8.")
    print(
        "Runtime ratio (batch=2, accumulation=4 / batch=8, accumulation=1): "
        f"{batch2_time / batch8_time:.2f}x"
    )
    print(
        "Metrics need not be identical: floating-point summation order and "
        "mini-batch-dependent layers can change updates. This model has no "
        "BatchNorm or dropout, so the gradient averages match closely when "
        "the same examples are grouped."
    )


if __name__ == "__main__":
    main()