import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Parameter configurations
EPOCHS = 300
LEARNING_RATE = 0.1
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