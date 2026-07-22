# YOLO-World 效果检测方式

你现在验效果分两层。

## 1. 先验接口

先确认这三个东西：

```text
输入 JSON 能不能读
类别名能不能转成 YOLO-World classes
输出 JSON 的字段是不是固定格式
```

运行：

```powershell
cd D:\code\oil-gas-vlm\downstream
python scripts\create_demo_seismic_image.py --output examples\demo_inline_120.png
python scripts\run_yolo_world_detection.py `
  --input examples\vlm_yolo_world_sample.json `
  --output-dir outputs\yolo_demo `
  --backend mock `
  --write-overlays `
  --validate
```

看输出里有没有：

- `records`
- `class_prompts`
- `detections`
- `overlay_svg`

## 2. 再看效果

如果只是 mock 后端，能看的只有：

- 类别有没有被保留
- ROI 有没有按 prompt 生效
- JSON 结构有没有稳定

运行评价脚本：

```powershell
python scripts\evaluate_yolo_world_detection.py `
  --input outputs\yolo_demo\yolo_world_detections.json
```

它会输出：

- `class_coverage`
- `roi_hit_rate`
- `mean_score`
- 每个样本的检测统计

## 3. 真正看 YOLO-World 效果

装好真实模型后再跑：

```powershell
python scripts\run_yolo_world_detection.py `
  --input examples\vlm_yolo_world_sample.json `
  --output-dir outputs\yolo_demo_real `
  --backend ultralytics-yolo-world `
  --model-path yolov8s-world.pt `
  --device cuda:0 `
  --write-overlays
```

然后再跑一次评价脚本：

```powershell
python scripts\evaluate_yolo_world_detection.py `
  --input outputs\yolo_demo_real\yolo_world_detections.json
```

## 4. 怎么判断“效果好不好”

先看三条：

1. `class_coverage` 是否明显高于 mock。
2. `roi_hit_rate` 是否稳定。
3. overlay 上的框是否真的落在断层、亮点、河道候选区附近。

如果后面有人工标注，再加 IoU / mAP 才算正式评测。

## 5. 没有检出框怎么办

先判断是阈值/ROI 过滤问题，还是模型本身对地震图不敏感：

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

然后查看输出 JSON 里的 `warnings`。如果看到 `YOLO raw boxes: 0`，说明模型在当前地震图和类别词下完全没有候选框；如果 raw boxes 大于 0 但 kept 为 0，说明是阈值、类别名或 ROI 过滤导致的。
