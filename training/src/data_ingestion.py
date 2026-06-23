import os
import boto3


def download_dataset_from_s3(
    bucket_name: str,# "maaz-animal-classifier-v3"
    s3_prefix: str,  # "data/raw/dataset"
    local_dir: str   # "/tmp/data"
):
    """
    Download dataset recursively from S3.
    """

    s3_client = boto3.client("s3")

    paginator = s3_client.get_paginator("list_objects_v2")

    pages = paginator.paginate(
        Bucket=bucket_name,
        Prefix=s3_prefix
    )

    """
        pages : 
            {
                "Contents": [

                    {
                        "Key": "data/raw/dataset/Cat/1.jpg"
                    },
                    ....
                ]
            }
    """
    for page in pages:

        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            # obj :  {"Key": "data/raw/dataset/Cat/1.jpg"  },

            s3_key = obj["Key"]

            if s3_key.endswith("/"): # we can file 1.jpg not 1/
                continue

            relative_path = os.path.relpath(
                s3_key,   # "data/raw/dataset/Cat/1.jpg"
                s3_prefix # "data/raw/dataset"
            )
            # relatvie_path : "Cat/1.jpg"

            local_file_path = os.path.join(
                local_dir,    # "/tmp/data"
                relative_path # "Cat/1.jpg"
            )
            # local_file_path : "/tmp/data/Cat/1.jpg"

            os.makedirs(
                os.path.dirname(local_file_path),
                exist_ok=True
            )

            print(
                f"Downloading: {s3_key}"
            )

            s3_client.download_file(
                bucket_name,    # "maaz-animal-classifier-v3"
                s3_key,         # "data/raw/dataset/Cat/1.jpg"
                local_file_path # "/tmp/data/Cat/1.jpg"
            )

    print(
        f"\nDataset downloaded to: {local_dir}"
    )



# ######## What is list_objects_v2() ?
# general aws api to list objects in a bucket.
# Think about S3.

# Remember:

# S3 has no real folders.

# It only stores:

# Object Keys

# Example:

# data/raw/dataset/Cat/1.jpg

# data/raw/dataset/Cat/2.jpg

# data/raw/dataset/Dog/1.jpg

# data/raw/dataset/Dog/2.jpg

# When we call:

# response = s3_client.list_objects_v2(
#     Bucket="maaz-animal-classifier-v3",
#     Prefix="data/raw/dataset"
# )

# AWS searches:

# Give me all object keys
# starting with

# data/raw/dataset
# What Does It Return?

# Something similar to:

# {
#     "Contents": [

#         {
#             "Key": "data/raw/dataset/Cat/1.jpg"
#         },

#         {
#             "Key": "data/raw/dataset/Cat/2.jpg"
#         },

#         {
#             "Key": "data/raw/dataset/Dog/1.jpg"
#         },

#         {
#             "Key": "data/raw/dataset/Dog/2.jpg"
#         }

#     ]
# }

# Notice:

# There is NO:

# {
#     "Folder": "Cat"
# }

# or

# {
#     "Folder": "Dog"
# }

# because folders don't exist.

# What Will We Get In Our Case?

# You uploaded:

# data/raw/dataset/Cat/....
# data/raw/dataset/Dog/....

# Therefore AWS will return thousands of entries like:

# {
#     "Key": "data/raw/dataset/Cat/1.jpg"
# }

# {
#     "Key": "data/raw/dataset/Cat/2.jpg"
# }

# {
#     "Key": "data/raw/dataset/Cat/3.jpg"
# }

# ...

# and

# {
#     "Key": "data/raw/dataset/Dog/1.jpg"
# }

# {
#     "Key": "data/raw/dataset/Dog/2.jpg"
# }





# ############# How are we going inside Cat and Dog folders?

# This is the most important realization:

# We are NOT.

# 😄

# Because:

# Cat
# Dog

# are not real folders.

# We simply iterate over:

# page["Contents"]

# which already contains:

# Cat Images
# +
# Dog Images

# Visual:

# AWS returns:

# [
#     "data/raw/dataset/Cat/1.jpg",
#     "data/raw/dataset/Cat/2.jpg",
#     "data/raw/dataset/Cat/3.jpg",

#     "data/raw/dataset/Dog/1.jpg",
#     "data/raw/dataset/Dog/2.jpg",
#     "data/raw/dataset/Dog/3.jpg"
# ]

# Then:

# for obj in page["Contents"]:

# loops through:

# Cat Image 1
# Cat Image 2
# Cat Image 3
# Dog Image 1
# Dog Image 2
# Dog Image 3
# ...

# one by one.

# The Magic Line

# This line:

# relative_path = os.path.relpath(
#     s3_key,
#     s3_prefix
# )

# Suppose:

# s3_key
# =
# "data/raw/dataset/Cat/123.jpg"

# and

# s3_prefix
# =
# "data/raw/dataset"

# Result:

# relative_path
# =
# "Cat/123.jpg"

# Similarly:

# s3_key
# =
# "data/raw/dataset/Dog/999.jpg"

# becomes:

# relative_path
# =
# "Dog/999.jpg"

# Then:

# local_file_path = os.path.join(
#     local_dir,
#     relative_path
# )

# creates:

# /tmp/data/Cat/123.jpg

# /tmp/data/Dog/999.jpg
# Visual Flow

# S3:

# data/raw/dataset/Cat/1.jpg
# data/raw/dataset/Cat/2.jpg
# data/raw/dataset/Dog/1.jpg
# data/raw/dataset/Dog/2.jpg

# Loop:

# for obj in page["Contents"]

# Creates:

# /tmp/data/Cat/1.jpg
# /tmp/data/Cat/2.jpg
# /tmp/data/Dog/1.jpg
# /tmp/data/Dog/2.jpg
# Mentor Question

# Suppose tomorrow your bucket contains:

# data/raw/dataset/Cat/...

# data/raw/dataset/Dog/...

# data/raw/dataset/Tiger/...

# data/raw/dataset/Zebra/...

# Will we need to change:

# download_dataset_from_s3()

# at all?

# Or will it automatically download everything?