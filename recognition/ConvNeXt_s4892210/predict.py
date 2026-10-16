import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from dataset import ADNI_MEAN, ADNI_STD
from modules import build_model

# Parameter configurations
OUTPUT_DIR = "assets"
BATCH_SIZE = 256
BASELINE_CHECKPOINT = os.path.join(OUTPUT_DIR, "best_baseline.pth")
CONVNEXT_CHECKPOINT = os.path.join(OUTPUT_DIR, "best_convnext.pth")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def load_model(model_name, checkpoint_path):
    """

    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()

    return model


def count_parameters(model):
    """

    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def run_inference(model, dataloader):
    """

    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_images = []

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)

    warmup_complete = False
    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Warm up CUDA before measuring latency.
            if DEVICE.type == "cuda" and not warmup_complete:
                for _ in range(5):
                    _ = model(images)
                torch.cuda.synchronize()
                warmup_complete = True

            outputs = model(images)

            probabilities = F.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

            # Store CPU images for example figures.
            all_images.append(images.cpu().numpy())

    probabilities = np.concatenate(all_probabilities, axis=0)
    images = np.concatenate(all_images, axis=0)
    latency = measure_inference_latency(model, dataloader)

    if DEVICE.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(DEVICE)
    else:
        peak_vram = 0

    return (
        np.array(all_predictions),
        np.array(all_labels),
        probabilities,
        images,
        latency,
        peak_vram
    )


def measure_inference_latency(model, dataloader):
    """

    """

    model.eval()
    total_time = 0.0
    total_images = 0

    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(dataloader):
            images = images.to(DEVICE)
            if batch_index >= 5:
                break

            _ = model(images)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    with torch.inference_mode():
        for images, _ in dataloader:
            images = images.to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            _ = model(images)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            total_time += end_time - start_time
            total_images += images.size(0)

    latency_ms_per_image = (total_time / total_images) * 1000.0
    return latency_ms_per_image


def calculate_metrics(predictions, labels, probabilities):
    """

    """

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    ad_precision = precision_score(labels, predictions, pos_label=0, zero_division=0)
    ad_recall = recall_score(labels, predictions, pos_label=0, zero_division=0)
    nc_precision = precision_score(labels, predictions, pos_label=1, zero_division=0)
    nc_recall = recall_score(labels, predictions, pos_label=1, zero_division=0)

    ad_labels = (labels == 0).astype(int)
    ad_probabilities = probabilities[:, 0]
    auroc = roc_auc_score(ad_labels, ad_probabilities)
    cm = confusion_matrix(labels, predictions)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "auroc": auroc,
        "ad_precision": ad_precision,
        "ad_recall": ad_recall,
        "nc_precision": nc_precision,
        "nc_recall": nc_recall,
        "confusion_matrix": cm,
    }


def print_results(model_name, model, metrics, latency_ms, peak_vram):
    """

    """

    display_name = "Baseline CNN" if model_name == "baseline" else "ConvNeXt"
    parameter_count = count_parameters(model)

    print(f"\n=== {display_name} Test Results ===")
    print(f"Test Accuracy       : {metrics['accuracy']:.4f}")
    print(f"F1-Score            : {metrics['macro_f1']:.4f}")
    print(f"AUROC               : {metrics['auroc']:.4f}")
    print(f"AD Precision        : {metrics['ad_precision']:.4f}")
    print(f"AD Recall           : {metrics['ad_recall']:.4f}")
    print(f"NC Precision        : {metrics['nc_precision']:.4f}")
    print(f"NC Recall           : {metrics['nc_recall']:.4f}")
    print(f"Parameters          : {parameter_count:,}")
    print(f"Inference Latency   : {latency_ms:.3f} ms/image")

    if DEVICE.type == "cuda":
        print(f"Peak GPU VRAM      : {peak_vram / (1024 ** 3):.3f} GB")
    else:
        print("Peak GPU VRAM       : N/A (CPU)")


def plot_confusion_matrix(model_name, confusion_matrix_values):
    """

    """

    display_name = "Baseline CNN" if model_name == "baseline" else "ConvNeXt"

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_values,
        display_labels=["AD", "NC"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"{display_name} Test Confusion Matrix")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_roc_curve(labels, probabilities):
    """

    """
    ad_labels = (labels == 0).astype(int)
    ad_probabilities = probabilities[:, 0]
    false_positive_rate, true_positive_rate, _ = roc_curve(ad_labels, ad_probabilities)
    auroc = roc_auc_score(ad_labels, ad_probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ConvNeXt (AUROC = {auroc:.4f})"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ConvNeXt ROC Curve")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "convnext_roc_curve.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_accuracy_vs_confidence(labels, predictions, probabilities):
    """

    """
    confidence = probabilities.max(axis=1)
    num_bins = 10
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = []
    bin_accuracies = []

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = ((confidence >= lower) & (confidence < upper))

        if upper == 1.0:
            mask = ((confidence >= lower) & (confidence <= upper))

        if np.sum(mask) == 0:
            continue

        bin_centers.append((lower + upper) / 2)
        bin_accuracies.append(np.mean(predictions[mask] == labels[mask]))

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, bin_accuracies, marker="o")
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Accuracy")
    plt.title("ConvNeXt Accuracy vs Confidence")
    plt.xlim(0.5, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "convnext_accuracy_vs_confidence.png")
    plt.savefig(path, dpi=200)
    plt.close()


def select_examples(
    labels,
    predictions,
    probabilities,
    correct=True,
    num_examples=9
):
    """

    """
    confidence = probabilities.max(axis=1)

    if correct:
        candidate_indices = np.where(predictions == labels)[0]
    else:
        candidate_indices = np.where(predictions != labels)[0]

    if len(candidate_indices) == 0:
        return []

    confidence_ranges = [
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 1.01),
    ]

    selected = []
    if num_examples == 9:
        target_per_range = [2, 2, 2, 2, 1]
    else:
        target_per_range = [1] * len(confidence_ranges)

    for ((lower, upper), target_count) in zip(confidence_ranges, target_per_range):
        candidates = candidate_indices[
            (confidence[candidate_indices] >= lower) &
            (confidence[candidate_indices] < upper)
        ]

        candidates = candidates[np.argsort(confidence[candidates])]

        for index in candidates[:target_count]:
            selected.append(index)

    if len(selected) < num_examples:
        remaining = [index for index in candidate_indices if index not in selected]
        remaining = sorted(remaining, key=lambda index: confidence[index])

        for index in remaining:
            selected.append(index)
            if len(selected) >= num_examples:
                break

    return selected[:num_examples]


def unnormalize_image(image):
    """

    """
    image = image[0]
    image = image * ADNI_STD + ADNI_MEAN
    image = np.clip(image, 0.0, 1.0)
    return image


def plot_prediction_examples(
    model_name,
    labels,
    predictions,
    probabilities,
    images,
    correct=True
):
    """

    """
    if correct:
        num_examples = 9
        title = "Correct Predictions"
        filename = (f"{model_name}_correct_predictions.png")
    else:
        num_examples = 5
        title = "Failure Predictions"
        filename = (f"{model_name}_failure_predictions.png")

    indices = select_examples(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        correct=correct,
        num_examples=num_examples
    )

    if len(indices) == 0:
        print(f"No {'correct' if correct else 'failure'} examples found for {model_name}.")
        return

    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    class_names = {0: "AD", 1: "NC"}
    for plot_index, data_index in enumerate(indices):
        row = plot_index // cols
        col = plot_index % cols
        ax = axes[row, col]
        image = unnormalize_image(images[data_index])
        prediction = class_names[predictions[data_index]]
        true_label = class_names[labels[data_index]]
        confidence = probabilities[data_index].max()

        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"Prediction: {prediction}\n"
            f"True: {true_label}\n"
            f"Confidence: {confidence * 100:.1f}%",
            fontsize=11
        )
        ax.axis("off")

    display_name = "Baseline CNN" if model_name == "baseline" else "ConvNeXt"

    fig.suptitle(f"{display_name} - {title}", fontsize=16)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()