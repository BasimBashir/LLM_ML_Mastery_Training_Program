import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Compare FP32, FP16, and BF16 training runtime, VRAM, and validation loss."
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("day6_amp_results.csv"),
        help="CSV file used to record the comparison (default: day6_amp_results.csv).",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in the range [0, 1).")
    if args.max_features < 2:
        parser.error("--max-features must be at least 2.")

    train_script = Path(__file__).with_name("train.py")
    results = []
    with tempfile.TemporaryDirectory(prefix="day6_amp_") as temp_dir:
        for precision in ("fp32", "fp16", "bf16"):
            checkpoint = Path(temp_dir) / f"{precision}.pt"
            command = [
                sys.executable,
                str(train_script),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--warmup-ratio",
                str(args.warmup_ratio),
                "--max-features",
                str(args.max_features),
                "--seed",
                str(args.seed),
                "--precision",
                precision,
                "--checkpoint",
                str(checkpoint),
            ]
            print(f"\nRunning {precision.upper()}...")
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                print(result.stdout, end="")
                print(result.stderr, end="", file=sys.stderr)
                result.check_returncode()

            print(result.stdout, end="")
            metrics_lines = [
                line.partition(": ")[2]
                for line in result.stdout.splitlines()
                if line.startswith("Experiment metrics: ")
            ]
            if len(metrics_lines) != 1:
                raise RuntimeError(
                    f"Expected one metrics record from {precision} training; "
                    f"found {len(metrics_lines)}."
                )
            results.append(json.loads(metrics_lines[0]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "precision",
        "peak_vram_allocated_mb",
        "peak_vram_reserved_mb",
        "training_samples_per_second",
        "validation_loss",
        "training_seconds",
    )
    with args.output.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n| Mode | Peak VRAM allocated (MB) | Peak VRAM reserved (MB) | Speed (samples/s) | Validation loss |")
    print("| ---- | -----------------------: | ----------------------: | ----------------: | --------------: |")
    for result in results:
        allocated = result["peak_vram_allocated_mb"]
        reserved = result["peak_vram_reserved_mb"]
        print(
            f"| {result['precision'].upper()} "
            f"| {allocated if allocated is not None else 'N/A'} "
            f"| {reserved if reserved is not None else 'N/A'} "
            f"| {result['training_samples_per_second']:.2f} "
            f"| {result['validation_loss']:.6f} |"
        )
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
