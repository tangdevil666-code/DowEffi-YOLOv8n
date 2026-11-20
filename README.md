# DowEffi-YOLOv8n

Code-release staging repository for *A lightweight row-direction-aware framework for maize seedling detection and plot-scale counting from UAV RGB imagery*.

## Release status

**Code may be published; the supplied paper weight is a local Release asset pending independent load validation.** The complete YOLOv8 code and model YAML were recovered from a separate local supplemental package. The supplied `best.pt` is 5,010,601 bytes with SHA-256 `50046cc8cabaa0dc2de6e2c39330b9f705fd43e931e0dab061e72095a1a2c31a`; its ZIP structure contains an `EfficientHead` marker. It is deliberately Git-ignored and needs an independent `torch.load`, architecture build, and output-equivalence check before public upload.

The experiment snapshot is kept read-only. This repository contains `EfficientHead`, module registration, NWD loss, release scaffolding and integrity checks. It contains no field imagery, labels, manifests, weights, or private source-path mapping.

## Method and model boundary

The paper model is **YOLOv8n**, not YOLO26. The intended full model uses `EfficientHead` and NWD loss (`C=12.8`) with the training gains recorded by the source run: box/NWD 7.5, classification 0.5, DFL 1.5, 100 epochs, image size 640, and seed 0. These are provenance records, not evidence of a valid released checkpoint.

`EfficientHead` is implemented in `ultralytics/nn/modules/head.py`; parser registration is in `ultralytics/nn/tasks.py`; NWD is in `ultralytics/utils/loss.py`. In the supplied source, `BboxLoss` was globally changed to NWD. Before release, baseline training must be restored to CIoU and NWD must be selected only by a model-specific configuration/criterion.

## Data access

UAV imagery and complete labels are restricted and are not included. The fixed internal split, LOYO folds, and Manas external manifest have not yet been recovered from a complete data source. Do not replace them with a fresh random split. Once authorized data are available, use `scripts/prepare_data.py` to generate a local data YAML without copying data into this repository.

## Integrity check

```powershell
python scripts/check_release.py --source PATH_TO_READONLY_EXPERIMENT_SOURCE
```

The command exits non-zero while required files are empty or truncated. After a clean source snapshot is supplied, create an isolated environment and install this fork with `python -m pip install -e .`; do not install or upgrade the PyPI `ultralytics` package over it.

## Required before publication

1. Re-copy `runs/detect/huizong/train_EfficientHead_NWDLoss/yolov8n_EfficientHead_NWDLoss/weights/best.pt` and verify its SHA-256 and `torch.load` in an isolated environment.
3. Recover actual fixed, LOYO, and Manas manifests; publish only opaque IDs where filenames are restricted.
4. Separate baseline CIoU and full-model NWD criteria, then run import, model-build, backward, and single-image checks.
5. Create `CITATION.cff`, `THIRD_PARTY_NOTICES.md`, model-zoo metadata, and release assets only from verified artifacts.

## License

This fork retains the source `LICENSE` (AGPL-3.0). Any publication must retain upstream notices and identify the local modifications.
