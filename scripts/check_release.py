"""Fail-fast audit for the locally supplied paper-release source snapshot."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REQUIRED_RELATIVE = (
    "ultralytics/cfg/models/v8/yolov8_EfficientHead_NWDLoss.yaml",
    "ultralytics/nn/modules/head.py",
    "ultralytics/nn/tasks.py",
    "ultralytics/utils/loss.py",
    "runs/detect/huizong/train_EfficientHead_NWDLoss/yolov8n_EfficientHead_NWDLoss/weights/best.pt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    for relative in REQUIRED_RELATIVE:
        path = args.source / relative
        if not path.is_file():
            failures.append(f"MISSING  {relative}")
        elif path.stat().st_size == 0:
            failures.append(f"EMPTY    {relative}")
        else:
            print(f"OK       {relative} ({path.stat().st_size} bytes)")
    checkpoint = args.source / REQUIRED_RELATIVE[-1]
    if checkpoint.is_file() and checkpoint.stat().st_size:
        print(f"SHA256   {sha256(checkpoint)}")
        if checkpoint.stat().st_size < 1024 * 1024:
            failures.append("TRUNCATED checkpoint is below 1 MiB; do not unpickle or publish it")
    if failures:
        print("\nFAILED RELEASE-SOURCE AUDIT:")
        print("\n".join(failures))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
