# YOLO-World 下游检测模块

这个子目录是 `oil-gas-vlm` 工程里的下游检测模块。

它只负责：

```text
蒋宇翔 VLM 输出的 downstream_prompts.yolo_world
  -> YOLO-World text prompts / classes
  -> 检测框 bbox
  -> 固定格式 JSON
```

## 快速运行

如果使用仓库自带的 demo 输入，先生成一张测试切片图：

```powershell
cd D:\code\oil-gas-vlm\downstream
python scripts\create_demo_seismic_image.py --output examples\demo_inline_120.png
```

然后跑 mock：

```powershell
cd D:\code\oil-gas-vlm\downstream
python scripts\run_yolo_world_detection.py `
  --input examples\vlm_yolo_world_sample.json `
  --output-dir outputs\yolo_demo `
  --backend mock `
  --write-overlays `
  --validate
```

评价输出：

```powershell
python scripts\evaluate_yolo_world_detection.py `
  --input outputs\yolo_demo\yolo_world_detections.json
```

## 真实 YOLO-World 后端

服务器安装好 YOLO-World / Ultralytics 权重后，把 `mock` 改成真实后端：

```powershell
python scripts\run_yolo_world_detection.py `
  --input examples\vlm_yolo_world_sample.json `
  --output-dir outputs\yolo_demo_real `
  --backend ultralytics-yolo-world `
  --model-path yolov8s-world.pt `
  --device cuda:0 `
  --write-overlays
```

如果真实模型没有检出任何框，先用调试参数降低阈值并关闭 ROI 过滤：

```powershell
python scripts\run_yolo_world_detection.py `
  --input examples\vlm_yolo_world_sample.json `
  --output-dir outputs\yolo_demo_real_debug `
  --backend ultralytics-yolo-world `
  --model-path yolov8s-world.pt `
  --device cuda:0 `
  --conf-override 0.01 `
  --disable-roi-filter `
  --write-overlays `
  --validate
```

## 目录

```text
adapters/yolo_world_adapter.py       VLM 输出到 YOLO-World 的适配层
adapters/sam_adapter.py              YOLO bbox / SAM prompt 到 mask 的适配层
schemas/yolo_world_schema.py         输出 JSON schema
scripts/run_yolo_world_detection.py  检测入口
scripts/evaluate_yolo_world_detection.py 轻量效果检查
examples/vlm_yolo_world_sample.json  蒋宇翔 VLM 输出样例
docs/YOLO_WORLD_ADAPTER.md           接口说明
```

## SAM 接口占位

现在还没有真实切片和标注，可以先把 SAM/SAM3 的输入输出合同补好：

```powershell
python scripts\run_sam_segmentation.py `
  --input outputs\yolo_demo\yolo_world_detections.json `
  --output-dir outputs\sam_demo `
  --backend mock `
  --write-overlays `
  --validate
```

说明见：

```text
docs/SAM_ADAPTER.md
```

## 一条命令串起 YOLO + SAM

```powershell
python scripts\run_downstream_pipeline.py `
  --input examples\vlm_yolo_world_sample.json `
  --output-dir outputs\pipeline_demo `
  --yolo-backend mock `
  --sam-backend mock `
  --write-overlays `
  --validate
```

说明见：

```text
docs/DOWNSTREAM_PIPELINE.md
```
