import os
import shutil
import tarfile

import requests


def download_and_extract_dataset(url, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    tar_path = os.path.join(extract_path, "dataset.tar.gz")
    # Download dataset
    r = requests.get(url, stream=True)
    with open(tar_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=128):
            f.write(chunk)
    # Extract dataset
    with tarfile.open(tar_path) as tar_ref:
        tar_ref.extractall(extract_path)
    os.remove(tar_path)  # Clean up tar file


def prepare_dataset(output_path):
    mali_path = os.path.join(
        output_path, "BreaKHis_v1/histology_slides/breast/malignant/"
    )
    ben_path = os.path.join(output_path, "BreaKHis_v1/histology_slides/breast/benign/")

    for path, kind in [(ben_path, "benign"), (mali_path, "malignant")]:
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".png"):
                    original = os.path.join(dirname, filename)
                    target_dir = os.path.join(output_path, "dataset", kind)
                    os.makedirs(target_dir, exist_ok=True)
                    target = os.path.join(target_dir, filename)
                    shutil.copyfile(original, target)


if __name__ == "__main__":
    url = "https://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
    output_path = "./working"
    download_and_extract_dataset(url, output_path)
    prepare_dataset(output_path)
