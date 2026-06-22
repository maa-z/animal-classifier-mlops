import json
import os

import torch
import torch.nn as nn

from torchvision import models


def load_model(
    artifact_dir,
    device
):

    # -------------------------
    # Load Config
    # -------------------------

    with open(
        os.path.join(
            artifact_dir,
            "config.json"
        ),
        "r"
    ) as f:

        config = json.load(f)

    # -------------------------
    # Load Classes
    # -------------------------

    with open(
        os.path.join(
            artifact_dir,
            "class_mapping.json"
        ),
        "r"
    ) as f:

        class_mapping = json.load(f)

    num_classes = len(class_mapping)

    model_name = (
        config["training"]["model_name"]
    )

    # -------------------------
    # Recreate Model
    # -------------------------
    model = None
    if model_name == "resnet18":

        model = models.resnet18(
            weights=None
        )

        in_features = (
            model.fc.in_features
        )

        model.fc = nn.Linear(
            in_features,
            num_classes
        )

    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    # -------------------------
    # Load Weights
    # -------------------------

    model_path = os.path.join(
        artifact_dir,
        "best_model.pth"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return (
        model,
        class_mapping,
        config
    )