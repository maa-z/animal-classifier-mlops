import os
import boto3

# load artifacts from s3 to local directory for inference
def download_artifacts(config):

    bucket_name = config["aws"]["bucket_name"]

    version = config["model"]["version"]
    s3_artifact_prefix = config["model"]["s3_artifact_prefix"]

    artifact_prefix = (
        f"{s3_artifact_prefix}/{version}"
    )

    local_artifact_dir = (
        config["model"]["local_artifact_dir"]
    )

    os.makedirs(
        local_artifact_dir,
        exist_ok=True
    )

    s3_client = boto3.client("s3")

    required_files = [
        "best_model.pth",
        "class_mapping.json",
        "config.json"
    ]

    for file_name in required_files:

        s3_key = (
            f"{artifact_prefix}/{file_name}"
        )

        local_path = os.path.join(
            local_artifact_dir,
            file_name
        )

        print(
            f"Downloading: {s3_key}"
        )

        s3_client.download_file(
            bucket_name,
            s3_key,
            local_path
        )

    return local_artifact_dir