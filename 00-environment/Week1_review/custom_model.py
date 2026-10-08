from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

DATA_DIR = Path(__file__).resolve().parent / "data"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "best.pt"
CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


class DeepMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1: Input 1x28x28 -> Output 32x28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: Input 32x28x28 -> Output 64x28x28 -> MaxPool -> 64x14x14
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: Input 64x14x14 -> Output 128x14x14 -> MaxPool -> 128x7x7
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling & Dropout Layer definitions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout(0.25) # 25% dropout for feature maps
        self.dropout_fc = nn.Dropout(0.5) # 50% dropout for dense

        # Fully Connected Layers (Input size: 128 channels * 7x7 image resolution)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10) # 10 classes for FashionMNIST


    def forward(self, x):
        # Layer1
        x = F.relu(self.bn1(self.conv1(x)))

        # Layer 2 + Pool + Dropout
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Layer 3 + Pool + Dropout
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 128 * 7 * 7)

        # Dense layer 1 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)

        # Output Layer (CrossEntropyLoss handles LogSoftmax automatically)
        x = self.fc2(x)
        return x


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


def evaluate(model, device, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * target.size(0)
            predictions = output.argmax(dim=1)
            correct += predictions.eq(target).sum().item()
            total += target.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    parser = ArgumentParser(description="Train a FashionMNIST CNN.")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    train_size = int(0.9 * len(training_data))
    validation_size = len(training_data) - train_size
    train_subset, validation_subset = torch.utils.data.random_split(
        training_data,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_subset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    best_validation_accuracy = -1.0

    print(f"Training on {device}")
    for epoch in range(1, args.epochs + 1):
        training_loss = train(model, device, train_loader, optimizer, criterion)
        validation_loss, validation_accuracy = evaluate(
            model, device, validation_loader, criterion
        )
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train loss: {training_loss:.4f}, "
            f"validation loss: {validation_loss:.4f}, "
            f"validation accuracy: {validation_accuracy:.2f}%"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "validation_accuracy": validation_accuracy,
                },
                CHECKPOINT_PATH,
            )
            print(f"Saved best checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()