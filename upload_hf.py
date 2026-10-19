"""
Upload release artifacts to Hugging Face Model Hub.
Reads token exclusively from the HF_TOKEN environment variable.
Never commits or logs the token.
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload power load forecasting benchmark artifacts to Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="neitpm/power-load-forecasting",
        help="Target repository ID on Hugging Face"
    )
    parser.add_argument(
        "--release_dir",
        type=str,
        default="hf_staging",
        help="Local release directory containing model artifacts and metadata"
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token or not token.strip():
        print("ERROR: HF_TOKEN environment variable is not set or empty.", file=sys.stderr)
        print("Please set your Hugging Face write token before running:", file=sys.stderr)
        print("  $env:HF_TOKEN = 'hf_...'", file=sys.stderr)
        sys.exit(1)

    release_path = Path(args.release_dir).resolve()
    if not release_path.exists():
        print(f"ERROR: Staging directory '{release_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to Hugging Face Hub with repository: {args.repo_id}...")
    api = HfApi(token=token.strip())

    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=False,
            exist_ok=True
        )
        print(f"Repository '{args.repo_id}' ready on Hugging Face Hub.")
    except Exception as e:
        print(f"Repository creation message / warning: {e}", file=sys.stderr)

    print(f"Uploading files from '{release_path}' to '{args.repo_id}'...")
    api.upload_folder(
        folder_path=str(release_path),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Release: reproducible power load forecasting KAN artifacts and benchmark manifests"
    )
    print("Upload completed successfully!")

if __name__ == "__main__":
    main()
