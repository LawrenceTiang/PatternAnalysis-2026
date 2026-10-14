import os
import time
import numpy as np
import torch
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