import os
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
