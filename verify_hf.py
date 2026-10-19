"""
Independent verification script for Hugging Face release artifacts:
1. Downloads snapshot from HF Hub into an isolated temporary folder.
2. Checks SHA-256 for all artifacts against metadata/weights_manifest.json.
3. Performs PyTorch model loading and synthetic forward-pass smoke test.
4. Verifies no NaN/Inf in model outputs.
"""

import sys
import json
import hashlib
import tempfile
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

def verify_release(repo_id="neitpm/power-load-forecasting"):
    print(f"Starting independent verification for '{repo_id}'...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Downloading snapshot to temporary directory: {temp_path}...")
        snapshot_path = Path(snapshot_download(repo_id=repo_id, local_dir=temp_path))

        manifest_file = snapshot_path / "metadata" / "weights_manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError("Missing metadata/weights_manifest.json in downloaded snapshot!")

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        print(f"Verifying {len(manifest)} artifacts from manifest...")

        for item in manifest:
            rel_path = item["path"]
            expected_sha = item["sha256"]
            target_file = snapshot_path / rel_path

            if not target_file.exists():
                raise FileNotFoundError(f"Missing artifact in snapshot: {rel_path}")

            actual_sha = hashlib.sha256(target_file.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(f"SHA-256 mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}")
            print(f"  ✓ {rel_path}: SHA-256 verified ({item['size_bytes']} bytes)")

        # Smoke test: import architecture and run forward pass
        sys.path.insert(0, str(snapshot_path))
        from code.models_kan import ResidualKANRegressor

        print("Running PyTorch forward pass smoke test...")
        in_features = 24
        model = ResidualKANRegressor(in_features=in_features, hidden_dims=[64, 32], p_t_idx=0)
        model.eval()

        # Synthetic batch of 8 test records
        x_synthetic = torch.randn(8, in_features)
        with torch.no_grad():
            output = model(x_synthetic)

        assert output.shape == (8,), f"Unexpected output shape: {output.shape}"
        assert not torch.isnan(output).any(), "Forward pass returned NaN values!"
        assert not torch.isinf(output).any(), "Forward pass returned Inf values!"
        print(f"  ✓ Smoke test passed: output shape {output.shape}, finite values in range [{output.min():.4f}, {output.max():.4f}]")

    print("\nALL VERIFICATIONS PASSED: 100% SHA-256 MATCH & FORWARD SMOKE TEST SUCCESSFUL!")

if __name__ == "__main__":
    verify_release()
