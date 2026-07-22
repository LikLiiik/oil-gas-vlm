# SAM / SAM3 下游分割接口

当前只实现 mock 接口，用于把数据合同先固定下来。

## 典型流程

```text
YOLO-World detections.json
  -> bbox prompts
  -> SAM/SAM3
  -> sam_masks.json
```

## 输入 1：直接 SAM prompt

```json
{
  "sample_id": "demo_sam_001",
  "image_path": "examples/demo_inline_120.png",
  "sam_prompts": [
    {
      "type": "bbox",
      "label": "fault plane",
      "bbox_xyxy": [140, 160, 430, 610]
    }
  ]
}
```

## 输入 2：YOLO 输出

可以直接把 `outputs/.../yolo_world_detections.json` 作为输入，脚本会自动把每个检测框转成 SAM 的 bbox prompt。

## 运行

```powershell
cd D:\code\oil-gas-vlm\downstream

python scripts\run_sam_segmentation.py `
  --input outputs\yolo_demo\yolo_world_detections.json `
  --output-dir outputs\sam_demo `
  --backend mock `
  --write-overlays `
  --validate
```

也可以跑直接输入：

```powershell
python scripts\run_sam_segmentation.py `
  --input examples\direct_sam_request.json `
  --output-dir outputs\sam_direct_demo `
  --backend mock `
  --write-overlays `
  --validate
```

评价：

```powershell
python scripts\evaluate_sam_segmentation.py `
  --input outputs\sam_demo\sam_masks.json
```

## 输出

```json
{
  "schema_version": "oil-gas.sam-segmentation.v1",
  "backend": {"name": "mock", "model_path": null, "device": "cpu"},
  "records": [
    {
      "sample_id": "demo_inline_120",
      "image_path": "examples/demo_inline_120.png",
      "prompts": [],
      "masks": [
        {
          "mask_id": "M1",
          "label": "fault plane",
          "score": 1.0,
          "polygon_xy": [[140,160],[430,160],[430,610],[140,610]],
          "bbox_xyxy": [140,160,430,610],
          "source": "mock_sam_mask"
        }
      ]
    }
  ]
}
```

后续接真实 SAM3 时，不改输出字段，只把 `mock_sam_mask` 替换成真实 mask。
