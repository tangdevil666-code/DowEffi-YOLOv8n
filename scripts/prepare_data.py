"""Write a local data YAML for authorized restricted data; never copies source data."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    for split in ("train", "val", "test"):
        if not (root / "images" / split).is_dir():
            raise SystemExit(f"missing images/{split} below {root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "path: " + root.as_posix() + "\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: maize\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
