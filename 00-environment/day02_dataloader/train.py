import argparse
import json
import math
import re  # Split article text into word-like tokens.
from collections import Counter  # Count how often each word appears.
from functools import partial  # Pre-fill arguments for reusable callbacks.
from pathlib import Path  # Build file paths safely across operating systems.
from time import perf_counter

import torch  # Tensor operations, device selection, and checkpoint I/O.
from dataset import load_dataset  # Load AG News samples as text and labels.
from torch import nn  # Neural-network layers and loss function.

TOKEN_PATTERN = re.compile(r"\b\w+\b")
NUM_CLASSES = 4  # AG News has four topic labels.
HIDDEN_SIZE = 256  # Number of learned features in each hidden layer.


def autocast_context(device, precision):
    """Return the autocast context for the selected training precision."""
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def tokenize(text):
    """Turn text into lowercase word tokens, ignoring punctuation."""
    return TOKEN_PATTERN.findall(text.lower())  # Normalize case, then extract words.


def build_vocabulary(dataloader, max_features):
    """Assign an integer ID to each of the most common training words.

    The vocabulary is learned from training data only. ID 0 is reserved for
    words the vocabulary does not contain, so actual words start at ID 1.
    """
    token_counts = Counter()  # Accumulate the frequency of every training word.
    for texts, _ in dataloader:  # Read a batch; labels are not needed for vocabulary.
        for text in texts:  # Process each article in this batch.
            token_counts.update(tokenize(text))  # Add this article's word counts.

    if not token_counts:
        raise ValueError("The training dataset contains no tokens.")

    return {  # Reserve index 0 for unknown words; known word IDs start at 1.
        token: index + 1
        for index, (token, _) in enumerate(token_counts.most_common(max_features - 1))
    }


def vectorize_text(text, vocabulary):
    """Represent one article as (word ID, occurrence count) pairs.

    Words missing from the vocabulary map to ID 0. This lets the model still
    count unseen words without growing its input size.
    """
    counts = Counter(tokenize(text))  # Count repeated words within this article.
    return [
        (vocabulary.get(token, 0), count)  # Map unknown words to the reserved ID 0.
        for token, count in counts.items()
    ]


def collate_bow_batch(batch, input_size):
    """Combine encoded articles into the dense tensors expected by the MLP.

    Each row is one article; each column stores the count for one word ID.
    """
    features = torch.zeros((len(batch), input_size), dtype=torch.float32)  # One word-count row per article.
    labels = torch.empty(len(batch), dtype=torch.long)  # Class IDs must be integer tensors.

    for row, (word_counts, label) in enumerate(batch):  # Fill one tensor row per sample.
        for index, count in word_counts:  # Visit each word present in this article.
            features[row, index] += count  # Store how many times this word occurs.
        labels[row] = label  # Store the article's correct topic.

    return features, labels  # Return the model inputs and their target classes.


class NeuralNetwork(nn.Module):
    """A multilayer perceptron that predicts one of four news topics."""

    def __init__(self, input_size):
        """Build the layers; input_size is the number of vocabulary IDs."""
        super().__init__()  # Initialize PyTorch's module bookkeeping.
        self.layers = nn.Sequential(  # Apply these layers in order during forward().
            nn.Linear(input_size, HIDDEN_SIZE),  # Convert word counts into hidden features.
            nn.ReLU(),  # Add non-linearity so the model can learn complex patterns.
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),  # Combine the learned hidden features.
            nn.ReLU(),  # Apply non-linearity again before classification.
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES),  # Produce one score for each topic.
        )

    def forward(self, x):
        """Convert a batch of word-count rows into four class scores."""
        return self.layers(x)  # Return four class scores for every article in the batch.


def train_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    device,
    gradient_accumulation_steps,
    precision,
    scaler,
):
    """Train for one epoch and return average loss and accuracy.

    An epoch means one complete pass through the training dataset. For each
    group of mini-batches, backpropagation accumulates gradients and the
    optimizer updates weights once.
    """
    model.train()  # Tell PyTorch this pass is for training, not evaluation.
    total_loss = 0.0  # Running sum used to calculate sample-average loss.
    correct = 0  # Number of correct predictions seen this epoch.
    samples = 0  # Number of articles processed so far.
    optimizer_steps = 0
    accumulated_samples = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_index, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)  # Move data to CPU or GPU.
        with autocast_context(device, precision):
            predictions = model(features)  # Ask the model to score each news topic.
            loss = loss_fn(predictions, labels)  # Compare predicted scores with true labels.
        scaler.scale(loss * labels.size(0)).backward()
        accumulated_samples += labels.size(0)

        batch_size = labels.size(0)  # Number of articles in this batch.
        # Multiply batch-average loss by its size to calculate a sample average.
        total_loss += loss.item() * batch_size  # Add this batch's total loss.
        correct += (predictions.argmax(dim=1) == labels).sum().item()  # Count right predictions.
        samples += batch_size  # Include this batch in the epoch-wide totals.

        is_accumulation_boundary = (batch_index + 1) % gradient_accumulation_steps == 0
        is_last_batch = batch_index + 1 == len(dataloader)
        if is_accumulation_boundary or is_last_batch:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_samples)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            update_skipped = scaler.is_enabled() and scaler.get_scale() < scale_before_step
            if not update_skipped:
                scheduler.step()
                optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples = 0

    if samples == 0:
        raise ValueError("The training dataloader produced no samples.")
    return total_loss / samples, correct / samples, optimizer_steps


def validation_loop(dataloader, model, loss_fn, device, precision):
    """Measure loss and accuracy without changing model parameters."""
    model.eval()  # Switch layers to evaluation behavior.
    total_loss = 0.0  # Accumulate loss so it can be averaged over all samples.
    correct = 0  # Count correct validation predictions.
    samples = 0  # Count validation articles processed.

    # Validation does not need gradients, saving memory and computation.
    with torch.no_grad():
        for features, labels in dataloader:  # Check each validation batch.
            features, labels = features.to(device), labels.to(device)  # Match the model's device.
            with autocast_context(device, precision):
                predictions = model(features)  # Predict without changing model weights.
                loss = loss_fn(predictions, labels)
            batch_size = labels.size(0)  # Track batch size to weight partial batches correctly.
            total_loss += loss.item() * batch_size  # Accumulate total loss.
            correct += (predictions.argmax(dim=1) == labels).sum().item()  # Count correct predictions.
            samples += batch_size  # Update the validation sample count.

    if samples == 0:
        raise ValueError("The validation dataloader produced no samples.")
    return total_loss / samples, correct / samples  # Report validation loss and accuracy.


def save_checkpoint(
    path, model, optimizer, scheduler, scaler, next_epoch, global_step, vocabulary, config
):
    """Save everything needed to continue training in a later process.

    Writing to a temporary file first and then replacing the destination
    prevents an interrupted save from leaving a partially written checkpoint.
    """
    path = Path(path)  # Normalize the destination as a filesystem path.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create missing checkpoint folders.
    temporary_path = path.with_name(path.name + ".tmp")  # Save separately before replacing.
    checkpoint = {  # Bundle training state into one file for the next run.
        # Model weights define predictions; optimizer state includes AdamW's
        # running averages, which are needed for its updates to continue properly.
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        # Progress and vocabulary make the resumed run use the same inputs.
        "next_epoch": next_epoch,
        "global_step": global_step,
        "vocabulary": vocabulary,
        "config": config,
        # Restore random-number generation for reproducible shuffling/operations.
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():  # Preserve random state for GPU-based training too.
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()  # Save each GPU's RNG state.

    try:
        torch.save(checkpoint, temporary_path)  # Write the complete checkpoint to a temporary file.
        temporary_path.replace(path)  # Replace the old checkpoint only after saving succeeds.
    finally:
        temporary_path.unlink(missing_ok=True)  # Remove a leftover temp file after an error.


def load_checkpoint(path):
    """Read a checkpoint onto the CPU before restoring it into a model."""
    return torch.load(path, map_location="cpu", weights_only=True)  # Load safe tensor/data contents on CPU.


def restore_training_state(checkpoint, model, optimizer, scheduler, scaler):
    """Copy saved model, optimizer, and random-number state into live objects."""
    model.load_state_dict(checkpoint["model_state_dict"])  # Restore learned model weights.
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # Restore AdamW's running update history.
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # Continue the learning-rate schedule.
    scaler.load_state_dict(checkpoint["scaler_state_dict"])  # Continue FP16 scale tracking.
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())  # Continue CPU random-number generation.
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])  # Continue GPU random-number generation.
    return checkpoint  # Give the caller access to saved epoch and batch progress.


def parse_args():
    """Read training settings supplied on the command line."""
    parser = argparse.ArgumentParser(description="Train an AG News bag-of-words MLP.")  # Set up CLI options.
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Total target epochs, including any epochs completed before resuming.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="Training precision; FP16 uses GradScaler on CUDA.",
    )
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("00-environment/day02_dataloader/checkpoints/ag_news_mlp.pt"),
    )
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint.")
    args = parser.parse_args()  # Read the options supplied when running the script.

    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.gradient_accumulation_steps < 1
        or args.max_features < 2
    ):
        parser.error(
            "--epochs, --batch-size, and --gradient-accumulation-steps must be positive; "
            "--max-features must be at least 2."
        )
    if args.learning_rate <= 0:  # A learning rate must move weights in a positive step size.
        parser.error("--learning-rate must be positive.")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative.")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in the range [0, 1).")
    if args.resume and not args.checkpoint.is_file():  # Resume requires an existing saved checkpoint.
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    return args  # Pass validated options back to main().


def main():
    """Prepare data, train the model, validate it, and save progress."""
    args = parse_args()  # Read the user's requested training settings.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Prefer GPU when available.
    if args.precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training requires a CUDA device; use --precision bf16 or fp32.")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 training was requested, but this CUDA device does not support BF16.")
    checkpoint = None  # New runs start without saved training state.

    if args.resume:
        checkpoint = load_checkpoint(args.checkpoint)  # Read the state from the previous run.
        vocabulary = checkpoint["vocabulary"]  # Reuse the exact same word-to-ID mapping.
    else:
        # Count training words once so every article maps to the same columns.
        raw_training_data = load_dataset(  # Load text batches before numeric conversion.
            split="train",
            batch_size=512,
            shuffle=False,
        )
        vocabulary = build_vocabulary(raw_training_data, args.max_features)  # Learn feature IDs from train text.

    # partial() supplies the shared vocabulary/input width to each batch.
    vectorizer = partial(vectorize_text, vocabulary=vocabulary)  # Reuse this mapping for every article.
    collate_fn = partial(collate_bow_batch, input_size=len(vocabulary) + 1)  # Reserve one extra column for ID 0.
    train_dataloader = load_dataset(  # Shuffle training examples to mix their order each epoch.
        split="train",
        transform=vectorizer,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    # The dataset's test split acts as validation data: it is not used for
    # gradient updates, only to report how well the model generalizes.
    validation_dataloader = load_dataset(  # Keep validation order stable; no weight updates happen here.
        split="test",
        transform=vectorizer,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    total_optimizer_steps = args.epochs * (
        (len(train_dataloader) + args.gradient_accumulation_steps - 1)
        // args.gradient_accumulation_steps
    )
    config = {
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_features": args.max_features,
        "total_optimizer_steps": total_optimizer_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "precision": args.precision,
    }
    if checkpoint is not None:
        saved_config = checkpoint["config"]
        if config != saved_config:
            raise ValueError(
                "Resume settings differ from the checkpoint. "
                f"Saved: {saved_config}; requested: {config}."
            )

    model = NeuralNetwork(input_size=len(vocabulary) + 1).to(device)  # Match input width and selected device.
    loss_fn = nn.CrossEntropyLoss()  # Compare the four topic scores with each correct class.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)

    def lr_multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = max(1, total_optimizer_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.precision == "fp16" and device.type == "cuda",
    )
    global_step = 0  # Count optimizer updates across all epochs.
    start_epoch = 0  # New runs begin at the first epoch.

    if checkpoint is not None:
        # Restore the model, AdamW moments, schedule, and random-number state.
        restore_training_state(checkpoint, model, optimizer, scheduler, scaler)
        start_epoch = checkpoint["next_epoch"]  # Continue after the last fully saved epoch.
        global_step = checkpoint["global_step"]  # Continue the optimizer-update counter.
        print(
            f"Resumed at epoch {start_epoch + 1}; optimizer and scheduler restored "
            f"({len(optimizer.state)} parameter states, optimizer step {global_step})."
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_seconds = 0.0
    last_validation_loss = None
    # The epoch count fixes the cosine schedule horizon and must match on resume.
    for epoch in range(start_epoch, args.epochs):  # Train only the epochs not already completed.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_started = perf_counter()
        train_loss, train_accuracy, optimizer_steps = train_loop(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            device,
            args.gradient_accumulation_steps,
            args.precision,
            scaler,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds += perf_counter() - train_started
        global_step += optimizer_steps
        validation_loss, validation_accuracy = validation_loop(  # Measure generalization after this epoch.
            validation_dataloader, model, loss_fn, device, args.precision
        )
        last_validation_loss = validation_loss
        print(  # Show both training progress and held-out validation performance.
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.2%} - "
            f"validation loss: {validation_loss:.4f}, "
            f"validation accuracy: {validation_accuracy:.2%} - "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}, optimizer steps: {global_step}"
        )
        # Save after validation so the checkpoint reflects the latest full epoch.
        save_checkpoint(  # Save immediately so completed work survives process interruption.
            args.checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
            next_epoch=epoch + 1,
            global_step=global_step,
            vocabulary=vocabulary,
            config=config,
        )
        print(f"Checkpoint saved to {args.checkpoint}")

    if start_epoch >= args.epochs:
        print(f"Checkpoint already reached the target of {args.epochs} epochs.")
    else:
        print("Done!")
        training_samples = len(train_dataloader.dataset) * (args.epochs - start_epoch)
        experiment_metrics = {
            "precision": args.precision,
            "validation_loss": last_validation_loss,
            "training_seconds": training_seconds,
            "training_samples_per_second": training_samples / training_seconds,
            "peak_vram_allocated_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
            "peak_vram_reserved_mb": (
                torch.cuda.max_memory_reserved(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
        }
        print(f"Experiment metrics: {json.dumps(experiment_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
