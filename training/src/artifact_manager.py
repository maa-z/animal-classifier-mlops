import os
import json
import boto3
import torch


def save_artifacts(
    model,
    classes,
    metrics,
    config
):

    artifact_dir = config["artifacts"]["local_dir"]  # /tmp/artifacts

    os.makedirs(
        artifact_dir,
        exist_ok=True
    )

    # -------------------
    # Model
    # -------------------

    model_path = os.path.join(  # /tmp/artifacts/best_model.pth
        artifact_dir,
        "best_model.pth"
    )

    torch.save(                  # model get saved in /tmp/artifacts with name best_model.pth
        model.state_dict(),
        model_path
    )

    # -------------------
    # Class Mapping
    # -------------------

    class_mapping = {
        str(idx): class_name
        for idx, class_name
        in enumerate(classes)
    }

    with open(
        os.path.join(
            artifact_dir,
            "class_mapping.json"
        ),
        "w"
    ) as f:

        json.dump(
            class_mapping,
            f,
            indent=4
        )

    # -------------------
    # Config
    # -------------------

    with open(
        os.path.join(
            artifact_dir,
            "config.json"
        ),
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    # -------------------
    # Metadata
    # -------------------

    metadata = {
        "best_accuracy":
        metrics["best_accuracy"],

        "model_name":
        config["training"]["model_name"],

        "epochs":
        config["training"]["epochs"]
    }

    with open(
        os.path.join(
            artifact_dir,
            "metadata.json"
        ),
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    return artifact_dir







def upload_artifacts_to_s3(
    artifact_dir,
    config
):

    bucket_name = (                       # "maaz-animal-classifier-v2"
        config["aws"]["bucket_name"]
    )

    s3_prefix = (                         # "artifacts/animal-classifier"  -> this is the folder inside the bucket where we want to upload our artifacts)
        config["artifacts"]["s3_prefix"]
    )

    s3_client = boto3.client("s3")

    for file_name in os.listdir(
        artifact_dir
    ):

        local_file = os.path.join(
            artifact_dir,
            file_name
        )

        s3_key = (
            f"{s3_prefix}/{file_name}"
        )

        print(
            f"Uploading {file_name}"
        )

        s3_client.upload_file(
            local_file,
            bucket_name,
            s3_key
        )