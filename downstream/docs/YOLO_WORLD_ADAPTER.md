# YOLO-World 下游检测适配层

这部分是你负责的下游模型模块。

它只做一件事：

```text
把蒋宇翔那边 VLM 已经输出的 yolo_world prompt
变成 YOLO-World 可执行的检测输入，
并输出固定格式的检测 JSON。
```

## 输入

输入既可以是直接请求，也可以是 VLM 的输出结果。

### 1. 直接请求

```json
{
  "sample_id": "demo_inline_120",
  "image_path": "examples/demo_inline_120.png",
  "classes": ["fault plane", "bright spot anomaly", "channel"]
}
```

### 2. VLM 输出

```json
{
  "sample_id": "demo_inline_120",
  "image_path": "examples/demo_inline_120.png",
  "coordinate_system": {
    "x_axis": "cdp",
    "x_range": [40, 220],
    "y_axis": "time_ms",
    "y_range": [500, 1800],
    "y_direction": "top_to_bottom"
  },
  "downstream_prompts": {
    "yolo_world": {
      "task": "open_vocabulary_detection",
      "categories": [
        {
          "class_name": "fault plane",
          "description": "同相轴垂直错断、反射终止",
          "expected_cdp_range": [60, 100],
          "expected_time_range_ms": [800, 1600],
          "confidence_threshold": 0.3,
          "max_detections": 5
        }
      ]
    }
  }
}
```

## 输出

统一输出到一个 JSON：

```json
{
  "schema_version": "oil-gas.yolo-world-detection.v1",
  "backend": {
    "name": "mock",
    "model_path": null,
    "device": "cpu",
    "imgsz": 640
  },
  "records": [
    {
      "sample_id": "demo_inline_120",
      "image_path": "examples/demo_inline_120.png",
      "image_size": {"width": 512, "height": 384},
      "prompt_source": "vlm_downstream_prompts",
      "coordinate_system": null,
      "class_prompts": [
        {
          "class_name": "fault plane",
          "description": "同相轴垂直错断、反射终止",
          "text_prompt": "fault plane",
          "confidence_threshold": 0.3,
          "max_detections": 5,
          "expected_cdp_range": [60, 100],
          "expected_time_range_ms": [800, 1600],
          "source_index": 0
        }
      ],
      "detections": [
        {
          "class_name": "fault plane",
          "score": 0.81,
          "bbox_xyxy": [120, 40, 220, 180],
          "class_index": 0,
          "roi_xyxy": [100, 20, 260, 220]
        }
      ],
      "overlay_svg": "outputs/yolo_demo/overlays/demo_inline_120_overlay.svg",
      "warnings": []
    }
  ],
  "warnings": []
}
```

## 运行方式

### 先跑 mock

```powershell
cd D:\code\oil-gas-LLM
python scripts\run_yolo_world_detection.py `
  --input examples\vlm_yolo_world_sample.json `
  --output-dir outputs\yolo_demo `
  --backend mock `
  --write-overlays
```

### 再切真实 YOLO-World

```powershell
python scripts\run_yolo_world_detection.py `
  --input examples\vlm_yolo_world_sample.json `
  --output-dir outputs\yolo_demo_real `
  --backend ultralytics-yolo-world `
  --model-path yolov8s-world.pt `
  --device cuda:0 `
  --imgsz 640 `
  --write-overlays
```

## 说明

- `class_name` 是给 YOLO-World 的核心文本类名。
- `description` 是蒋宇翔那边 VLM 给出的语义解释。
- `expected_cdp_range` / `expected_time_range_ms` 用来约束搜索区域。
- `confidence_threshold` 和 `max_detections` 是每个类自己的检测约束。
- 这个模块不做 planner，不做任务路由，只做 YOLO-World 检测适配。
