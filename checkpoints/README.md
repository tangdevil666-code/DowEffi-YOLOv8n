# Checkpoints

The supplied checkpoint is stored locally as a separate, Git-ignored release asset at `release_assets/weights/doweffi-yolov8n-seed0-best.pt`; its SHA-256 is in `release_assets/SHA256SUMS`. Its ZIP archive structure and `EfficientHead` pickle marker were checked. It is not committed to Git and still requires an independent `torch.load(..., map_location="cpu")`, architecture build, and prediction-equivalence check before public upload.
