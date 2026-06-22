import json
import torch

from src.data_ingestion import (
    download_dataset_from_s3
)

from src.dataset_loader import (
    create_dataloaders
)

from src.trainer import (
    create_model,
    train_model
)

from src.artifact_manager import (
    save_artifacts,
    upload_artifacts_to_s3
)


def load_config():

    with open("config.json", "r") as f:
        return json.load(f)


def main():

    config = load_config()

    print("Loading Config...")

    # -------------------------
    # Download Dataset
    # -------------------------

    # --------------------------------------------------------------------------------------------------------------------
    # don't run if already downloaded because it will download the dataset every time we run the training script and we don't want that
    
    print("Downloading Dataset...")
    download_dataset_from_s3(
        bucket_name=config["aws"]["bucket_name"],
        s3_prefix=config["data"]["dataset_prefix"],
        local_dir=config["data"]["local_dataset_dir"]
    )

    # -------------------------
    # DataLoaders
    # -------------------------

    print("Creating DataLoaders...")

    train_loader, val_loader, classes = (
        create_dataloaders(config)
    )

    # -------------------------
    # Device
    # -------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using Device: {device}")

    # -------------------------
    # Model
    # -------------------------

    model = create_model(
        num_classes=len(classes)
    )

    model = model.to(device)

    # -------------------------
    # Training
    # -------------------------

    print("Training Started...")

    best_model, metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # -------------------------
    # Save Artifacts
    # -------------------------

    print("Saving Artifacts...")

    artifact_dir = save_artifacts(
        model=best_model,
        classes=classes,
        metrics=metrics,
        config=config
    )

    # -------------------------
    # Upload Artifacts
    # -------------------------

    print("Uploading Artifacts To S3...")

    upload_artifacts_to_s3(
        artifact_dir=artifact_dir,
        config=config
    )

    print("Training Pipeline Completed Successfully")


if __name__ == "__main__":
    main()