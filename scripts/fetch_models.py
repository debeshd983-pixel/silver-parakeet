"""Fetches candidate models from Hugging Face Hub using pinned commit revisions (supply chain rule)."""
import argparse
import os
from huggingface_hub import snapshot_download

# Pinned commit revisions
PINNED_MODELS = {
    "classifier": {
        "repo_id": "Organika/sdxl-detector",
        # The previously pinned 657a8bf... does not exist on the Hub (404). This is
        # the real head revision. Always re-verify a pin before trusting it.
        "revision": "b37fede8562cb72b89ec201c0987f96ba21b518a",
        # NOT apache-2.0. The Hub reports cc-by-nc-3.0: non-commercial use only.
        "license": "cc-by-nc-3.0",
        # Prebuilt fp32 ONNX shipped in the repo; avoids a local torch export.
        "onnx_subpath": "onnx/model.onnx",
    },
    "clip": {
        "repo_id": "openai/clip-vit-base-patch16",
        "revision": "51a6c117565eb6399c5120cfd137b7858c27b738",
        "license": "mit",
    },
}


def fetch_models(output_dir: str = "models/raw"):
    os.makedirs(output_dir, exist_ok=True)
    for name, info in PINNED_MODELS.items():
        print(f"--> Fetching {name}: {info['repo_id']} at revision {info['revision'][:8]}...")
        dest = os.path.join(output_dir, name)
        snapshot_download(
            repo_id=info["repo_id"],
            revision=info["revision"],
            local_dir=dest,
            local_dir_use_symlinks=False,
            allow_patterns=["*.json", "*.onnx"],
        )
        print(f"Downloaded {name} to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch models with pinned commit hashes")
    parser.add_argument("--output-dir", default="models/raw", help="Directory to save raw model files")
    args = parser.parse_args()
    fetch_models(args.output_dir)
