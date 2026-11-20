# DowEffi-YOLOv8n

这是基于 Ultralytics AGPL-3.0 的 YOLOv8n 轻量 fork。主模型在 `configs/models/yolov8_EfficientHead_NWDLoss.yaml`：检测头为 `EfficientHead`，训练时由 `nwd: true` 启用 NWD（`C=12.8`）。`configs/models/yolov8n_baseline.yaml` 明确使用原始 CIoU，避免基线误用 NWD。

安装时请创建独立环境并执行 `python -m pip install -e .`；不要用 `pip install -U ultralytics` 覆盖本地代码。训练示例：

```text
yolo detect train model=configs/models/yolov8_EfficientHead_NWDLoss.yaml data=PATH_TO_DATA.yaml cfg=configs/train/doweffi_seed0.yaml
```

预测：`yolo detect predict model=PATH_TO_VERIFIED_WEIGHT source=PATH_TO_IMAGE imgsz=640`。

获得受限数据权限后，用 `scripts/prepare_data.py --data-root DATA_ROOT --output local_data.yaml` 创建本地数据配置。计数汇总使用 `scripts/count.py`，其会拒绝把缺失预测文件误当成零株。

已收到的 `best.pt` 已作为本地、Git 忽略的 Release asset 保存，SHA-256 为 `50046cc8cabaa0dc2de6e2c39330b9f705fd43e931e0dab061e72095a1a2c31a`。ZIP 归档结构和 `EfficientHead` 标识已通过检查，但在独立 `torch.load`、架构构建和输出一致性检查通过前，不应上传公开发布。影像、标注、真实文件名映射和私有路径均未公开。
