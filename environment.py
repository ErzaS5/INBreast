"""Print and persist the environment used in an experiment."""
from __future__ import annotations
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def environment_info() -> dict:
    import torch
    packages = ("torch", "torchvision", "timm", "numpy", "pandas", "pydicom", "Pillow",
                "scikit-image", "scikit-learn", "scipy", "matplotlib", "PyYAML", "pytest")
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(), "torch_threads": torch.get_num_threads(),
            "git_commit": git.stdout.strip() if git.returncode == 0 else None}


if __name__ == "__main__":
    print(json.dumps(environment_info(), indent=2))
