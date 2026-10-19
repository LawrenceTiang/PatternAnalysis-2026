import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import build_dataloader
from modules import build_model

# Parameter configurations
EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
WEIGHT_DECAY = 0.05
SEED = 42
OUTPUT_DIR = "assets"

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def train_one_epoch(model, dataloader, criterion, optimizer):
    """

    """
    model.train()
    running_loss = 0.0
    correct = 0
    num_samples = 0

    for images, targets in dataloader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = logits.max(1)
        correct += predicted.eq(targets).sum().item()
        num_samples += batch_size

    average_loss = running_loss / num_samples
    accuracy = correct / num_samples
    return average_loss, accuracy


def validate(model, dataloader, criterion):
    """

    """
    model.eval()
    running_loss = 0.0
    correct = 0
    num_samples = 0

    with torch.inference_mode():
        for images, targets in dataloader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()
            num_samples += batch_size

    average_loss = running_loss / num_samples
    accuracy = correct / num_samples
    return average_loss, accuracy


def plot_training_curves(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    model_name
):
    """

    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    display_name = "Baseline CNN" if model_name == "baseline" else "ConvNeXt"

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{display_name} Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    loss_path = os.path.join(OUTPUT_DIR, f"{model_name}_loss.png")
    plt.savefig(loss_path)
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{display_name} Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    accuracy_path = os.path.join(OUTPUT_DIR, f"{model_name}_accuracy.png")
    plt.savefig(accuracy_path)
    plt.close()


def train():
    """

    """
    parser = argparse.ArgumentParser(
        description="Train an ADNI image classification model."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "convnext"],
        default="convnext",
        help="Model architecture."
    )

    args = parser.parse_args()
    print(f"Using device: {DEVICE}")
    print(f"Model: {args.model}\n")

    train_loader, val_loader = build_dataloader(is_train=True, batch_size=BATCH_SIZE)
    model = build_model(args.model).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=EPOCHS
    )

    # Track metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )
        val_loss, val_acc = validate(model, val_loader, criterion)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            best_epoch = epoch

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model_path = os.path.join(OUTPUT_DIR, f"best_{args.model}.pth")
            torch.save(model.state_dict(), model_path)

            print(f"=== New Best Model (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}) =====\n")

    print(f"\n=== Best {args.model} Model ===")
    print(f"Best Epoch: {best_epoch}")
    print(f"Validation Loss: {best_val_loss:.4f}")
    print(f"Validation Accuracy: {best_val_accuracy:.4f}")

    # Plot results
    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        model_name=args.model
    )


if __name__ == "__main__":
    train()