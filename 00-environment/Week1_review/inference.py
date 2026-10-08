from argparse import ArgumentParser
from pathlib import Path

import torch
from custom_model import CHECKPOINT_PATH, CLASS_NAMES, DATA_DIR, DeepMNISTCNN
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


def main():
    parser = ArgumentParser(
        description="Evaluate the best FashionMNIST checkpoint on the test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Path to best.pt (default: next to this script).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="FashionMNIST download/cache directory.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(
            f"Checkpoint not found: {args.checkpoint}. "
            "Train the model first by running custom_model.py."
        )
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = DeepMNISTCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    test_data = datasets.FashionMNIST(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    class_correct = torch.zeros(len(CLASS_NAMES), device=device)
    class_total = torch.zeros(len(CLASS_NAMES), device=device)

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            total_loss += criterion(logits, labels).item()
            total_correct += predictions.eq(labels).sum().item()
            class_total += torch.bincount(labels, minlength=len(CLASS_NAMES))
            class_correct += torch.bincount(
                labels[predictions.eq(labels)], minlength=len(CLASS_NAMES)
            )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Test samples: {len(test_data)}")
    print(f"Test loss: {total_loss / len(test_data):.4f}")
    print(f"Test accuracy: {100.0 * total_correct / len(test_data):.2f}%")
    print("Per-class accuracy:")
    class_correct = class_correct.cpu().tolist()
    class_total = class_total.cpu().tolist()
    for index, class_name in enumerate(CLASS_NAMES):
        accuracy = 100.0 * class_correct[index] / class_total[index]
        print(f"  {class_name}: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
