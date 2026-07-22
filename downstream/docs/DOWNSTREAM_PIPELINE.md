# 下游检测-分割总 Pipeline

这个脚本把当前两个下游模块串起来：

```text
VLM 输出 JSON
  -> YOLO-World 检测
  -> SAM/SAM3 分割
  -> pipeline_summary.json
```

## Mock 全链路

```powershell
cd D:\code\oil-gas-vlm\downstream

python scripts\run_downstream_pipeline.py `
  --input examples\vlm_yolo_world_sample.json `
  --output-dir outputs\pipeline_demo `
  --yolo-backend mock `
  --sam-backend mock `
  --write-overlays `
  --validate
```

输出：

```text
outputs/pipeline_demo/yolo/yolo_world_detections.json
outputs/pipeline_demo/sam/sam_masks.json
outputs/pipeline_demo/pipeline_summary.json
```

## 真实 YOLO + Mock SAM

真实数据接上后，先跑这个组合最稳：

```powershell
python scripts\run_downstream_pipeline.py `
  --input examples\vlm_yolo_world_sample.json `
  --output-dir outputs\pipeline_real_yolo `
  --yolo-backend ultralytics-yolo-world `
  --yolo-model-path yolov8s-world.pt `
  --sam-backend mock `
  --device cuda:0 `
  --write-overlays `
  --validate
```

如果 YOLO 检不出框，可以调试：

```powershell
python scripts\run_downstream_pipeline.py `
  --input examples\vlm_yolo_world_sample.json `
  --output-dir outputs\pipeline_real_yolo_debug `
  --yolo-backend ultralytics-yolo-world `
  --yolo-model-path yolov8s-world.pt `
  --sam-backend mock `
  --device cuda:0 `
  --conf-override 0.01 `
  --disable-roi-filter `
  --write-overlays `
  --validate
```

## 只跑 YOLO

```powershell
python scripts\run_downstream_pipeline.py `
  --input examples\vlm_yolo_world_sample.json `
  --output-dir outputs\pipeline_yolo_only `
  --yolo-backend mock `
  --skip-sam `
  --validate
```

## 汇总文件

`pipeline_summary.json` 包含：

```json
{
  "outputs": {
    "yolo_detections": ".../yolo_world_detections.json",
    "sam_masks": ".../sam_masks.json"
  },
  "counts": {
    "samples": 1,
    "detections": 4,
    "masks": 4
  },
  "validation": {
    "yolo": {"ok": true, "errors": []},
    "sam": {"ok": true, "errors": []}
  }
}
```

