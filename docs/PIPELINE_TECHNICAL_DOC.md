# Oil-Gas VLM Pipeline 技术文档

> 版本: 2.0 | 日期: 2026-07-21

---

## 目录

1. [核心设计](#1-核心设计)
2. [统一闭环流程](#2-统一闭环流程)
3. [数据来源：geo_adapter](#3-数据来源geo_adapter)
4. [VLM 推理引擎](#4-vlm-推理引擎)
5. [Prompt 体系](#5-prompt-体系)
6. [下游模型注册表](#6-下游模型注册表)
7. [8 个下游模型](#7-8-个下游模型)
8. [输出与导出](#8-输出与导出)
9. [IO 模块](#9-io-模块)
10. [Schema 体系](#10-schema-体系)
11. [CLI 使用](#11-cli-使用)
12. [扩展指南](#12-扩展指南)
13. [文件索引](#13-文件索引)

---

## 1. 核心设计

### 1.1 设计理念

```
VLM = 大脑（场景理解 + 模型规划 + 地质验证）
下游 = 手（精确检测/追踪/分割/计算）

问题：VLM 擅长"这是什么、大致在哪"，不擅长精确坐标和数值
解法：VLM 只做规划和验证，精确执行交给下游，结果必须回环验证
```

### 1.2 三层架构

```
                        ┌──────────────────────────────────────┐
                        │        Qwen3-VL-8B (大脑)             │
                        │                                      │
                        │  ① 看到 5 张图 + 任务 + 8个模型清单    │
                        │  ② 规划: 选哪个模型、在哪个图上、用什么参数│
                        │  ③ 验证: 下游结果+原图 → 真? 假? 遗漏? │
                        │  ④ 迭代: 调整参数重试直到收敛          │
                        └──────┬──────────────┬────────────────┘
                               │ ② workflow    │ ③ results
                               ▼               ▼
               ┌───────────────────────────────────────────────────┐
               │              8 个下游模型 (手)                      │
               ├──────────┬──────────┬──────────┬──────────────────┤
               │seismic_  │horizon_  │facies_   │attribute_        │
               │domain_   │tracker   │classifier│extractor         │
               │model     │          │          │                  │
               ├──────────┼──────────┼──────────┼──────────────────┤
               │yolo_     │well_log_ │sam       │traditional_code  │
               │world     │analyzer  │          │                  │
               └──────────┴──────────┴──────────┴──────────────────┘
```

### 1.3 三条使用路径

| 路径 | CLI | 用途 | VLM角色 |
|------|-----|------|---------|
| **赛题主流程** | `--run-dir` | geo_adapter 前置 → 检测 → 3D SEG-Y | 自主规划+验证回环 |
| **Fallback SEG-Y** | `--input` | 直读 SEG-Y，逐切片检测 | 自主规划+验证回环 |
| **4-Agent 语义** | `--agent` | 独立语义解释，JSON报告 | 规划+单次执行 |

---

## 2. 统一闭环流程

赛题主流程(`run_from_adapter`)和 Fallback SEG-Y(`run_volume`)共享同一核心闭环：

### 2.1 五阶段流程

```
Phase 1: VLM 工作流规划
  ┌─────────────────────────────────────────────────────┐
  │ 输入:                                               │
  │   • 5 张图像 (inline/crossline/slice/patch/well)     │
  │   • 任务描述 (fault→seismic_domain_model ...)        │
  │   • 8 个下游模型完整能力清单                          │
  │   • 任务→模型推荐映射表 (TASK_MODEL_MAP)              │
  │                                                     │
  │ VLM 输出 JSON (WORKFLOW_PLAN_SCHEMA 约束):           │
  │   {                                                 │
  │     "scene_understanding": "...",                   │
  │     "workflow_steps": [                             │
  │       {                                            │
  │         "step": 1,                                 │
  │         "model": "seismic_domain_model",            │
  │         "image_name": "seismic_inline",             │
  │         "reason": "inline剖面适合断层属性检测",       │
  │         "instruction": {                           │
  │           "task": "fault_detection",                │
  │           "attribute": "gradient",                  │
  │           "confidence_threshold": 0.3               │
  │         }                                          │
  │       },                                           │
  │       {                                            │
  │         "step": 2,                                 │
  │         "model": "facies_classifier",               │
  │         "image_name": "seismic_slice",              │
  │         "reason": "时间切片适合沉积相平面聚类",       │
  │         "instruction": {"n_clusters": 5, ...}       │
  │       }                                            │
  │     ]                                              │
  │   }                                                │
  └─────────────────────────────────────────────────────┘
                              │
                              ▼
Phase 2: 执行下游模型
  ┌─────────────────────────────────────────────────────┐
  │ for step in workflow_steps:                         │
  │   image = image_by_name[step["image_name"]]          │
  │   model = downstream.get(step["model"])             │
  │   results = model.detect(instruction, image, context)│
  └─────────────────────────────────────────────────────┘
                              │
                              ▼
Phase 3: VLM 验证
  ┌─────────────────────────────────────────────────────┐
  │ 喂给 VLM: 原图 + 下游检测结果                        │
  │ VLM 逐条判断:                                       │
  │   ✓ 真断层(同相轴可见8ms错断)                        │
  │   ✗ 假阳性(河道边缘反射终止，非断层)                   │
  │   遗漏: CDP 115-125有微小错断未检测到                  │
  │                                                     │
  │ 输出:                                               │
  │   {verified: [{is_real: true/false, ...}],          │
  │    need_retry: true/false,                          │
  │    retry_instructions: {step:1, adjusted_params:{}}}│
  └─────────────────────────────────────────────────────┘
                              │
                              ▼
Phase 4: 迭代重试 (if need_retry)
  ┌─────────────────────────────────────────────────────┐
  │ 用 retry_instructions 更新对应 step 的 instruction    │
  │ → 回到 Phase 2                                       │
  │ 最多 max_iterations 轮 (默认 3)                      │
  └─────────────────────────────────────────────────────┘
                              │
                              ▼
Phase 5: 聚合输出
  ┌─────────────────────────────────────────────────────┐
  │ bbox_pixel → manifest.views.source_indices          │
  │ → 3D 坐标反变换 → 每类一个属性体 SEG-Y                │
  │ + 标注 PNG (bbox 叠加原图)                            │
  │ + report.json (汇总)                                 │
  └─────────────────────────────────────────────────────┘
```

### 2.2 关键代码路径

```python
# orchestrator.py — 赛题入口
def run_from_adapter(self, run_dir, out_dir, verify, max_iterations):
    pkg = load_run(run_dir)                    # ① 载入 geo_adapter 数据
    image_by_name = {im.name: im for im in pkg.images}

    # Phase 1: VLM 规划
    plan_resp = self.vlm.call_json(
        workflow_planning_prompt(),            # 系统提示词
        [im.pil for im in pkg.images],         # 5 张图
        self._build_competition_plan_text(),   # 任务+模型清单
        schema=WORKFLOW_PLAN_SCHEMA,           # 约束输出格式
    )
    steps = plan_resp.data["workflow_steps"]

    # Phase 2-3-4: 执行+验证+迭代
    for iteration in range(max_iterations):
        round_dets = self._execute_competition_steps(steps, image_by_name, pkg)
        if not verify: break
        ver_resp = self.vlm.call_json(VERIFICATION_PROMPT, images, ...)
        if not ver_resp.data["need_retry"]: break
        self._apply_competition_retry(steps, ver_resp.data)

    # Phase 5: 输出
    per_class = aggregate_adapter_detections(all_detections, manifest, shape)
    write_attribute_segy(vol, cube, out_path)  # 每类一个 SEG-Y
    export_annotated_png(result, image, ...)   # 标注 PNG
```

### 2.3 VLM 如何调用下游模型

```
VLM 输出 JSON                         代码执行
───────────                          ────────

"model": "seismic_domain_model"      downstream.get("seismic_domain_model")
                                         ↓
"instruction": {                      SeismicDomainDetector.detect(
  "task": "fault_detection",            instruction={"task":"fault_detection",...},
  "attribute": "gradient",              image=PIL.Image.open("inline_model.png"),
  "regions_of_interest": [...]          context={"array": numpy_2d_array}
}                                     )
                                         ↓
                                      返回 [{id, class_name, bbox_pixel, confidence}, ...]

"model": "horizon_tracker"            downstream.get("horizon_tracker")
                                         ↓
"instruction": {                      HorizonTracker.detect(
  "seed_points": [{...}],               instruction={...},
  "tracking_mode": "correlation"        image=PIL.Image.open("inline_model.png"),
}                                     )
                                         ↓
                                      返回 [{id, points:[{trace_idx, sample_idx, confidence}]}]
```

---

## 3. 数据来源：geo_adapter

### 3.1 前置处理链路

```
原始数据                               geo_adapter                         VLM 输入
────────                              ───────────                        ────────

.sgy 地震体 ──┐
              │                      ① 读 SEG-Y
.las 测井 ────┤                      ② 3D → 2D 切片 (4个视图)
              ├─ geo-adapter prepare ─▶ ③ 归一化 (clip 98% percentile)
.csv 井位 ────┤                      ④ matplotlib 渲染为 PNG
              │                      ⑤ 读 LAS → 6道测井综合图
.npz 数组 ────┘                      ⑥ 生成 prompt/manifest/schema

                                          输出: runs/<sample_id>/
                                            ├── assets/seismic/
                                            │   ├── inline_model.png       ① VLM看
                                            │   ├── crossline_model.png     ② VLM看
                                            │   ├── slice_model.png         ③ VLM看
                                            │   └── local_patch_model.png   ④ VLM看
                                            ├── assets/well_logs/
                                            │   └── well_log_panel.png      ⑤ VLM看
                                            ├── prompts/
                                            │   ├── system_prompt.txt
                                            │   └── user_prompt.txt
                                            ├── manifest.json
                                            ├── request.json
                                            └── schemas/expected_model_output.schema.json
```

### 3.2 五张图像

| image_name | view | 内容 | VLM 用途 |
|------------|------|------|----------|
| `seismic_inline` | inline | 沿主测线垂直剖面 | 断层/层位检测（垂向分辨率） |
| `seismic_crossline` | crossline | 沿联络线正交剖面 | 断层/层位交叉验证 |
| `seismic_slice` | time_or_depth_slice | 时间/深度水平切片 | 河道/沉积相平面展布 |
| `seismic_local_patch` | local_horizontal_patch | 井旁局部放大 | 井震标定细节 |
| `well_log_panel` | well_log_panel | GR/RT/DEN/CNL/SP 综合图 | 岩性+流体分析 |

### 3.3 RunPackage 加载

```python
# adapter.py:88-126
def load_run(run_dir) -> RunPackage:
    manifest = _read_json("manifest.json")        # shape/crs/views/alignment
    request  = _read_json("request.json")         # 消息结构 → 决定加载哪些图
    system_prompt = _read_text("prompts/system_prompt.txt")
    user_prompt   = _read_text("prompts/user_prompt.txt")

    images = []
    for msg in request["messages"]:               # 从 request 里抓 image 项
        for c in msg["content"]:
            if c["type"] == "image":
                img = Image.open(run_dir / c["path"])
                images.append(PackageImage(
                    name=c["name"],
                    path=img_path,
                    physical_view=c["physical_view"],
                    pil=img,
                ))
    return RunPackage(...)
```

`manifest.json` 中的 `seismic.views[view_name].source_indices` 记录了每个切片的 inline/crossline/sample 索引，用于最终 bbox 到 3D 坐标的反变换。

---

## 4. VLM 推理引擎

**文件**: `pipeline/vlm.py`

### 4.1 VLMClient

```python
class VLMClient:
    def __init__(self, model_path="Qwen3-VL-8B-Instruct", dtype="bfloat16",
                 device_map="auto"):
        self._model = None       # 惰性加载，首次调用时加载
        self._processor = None
```

模型: Qwen3-VL-8B-Instruct，支持多图输入，bfloat16 推理，device_map="auto"

### 4.2 底层推理 `call()`

```python
def call(self, system_prompt, images: list[PIL.Image], user_text,
         max_new_tokens=4096, temperature=0.0) -> tuple[str, float]:
    # 构建 message: system + user(content=[image×N, text])
    # 调用 model.generate()
    # 返回 (decoded_text, elapsed_seconds)
```

### 4.3 带 Schema 重试的 JSON 调用 `call_json()`

```python
def call_json(self, system_prompt, images, user_text,
              schema=None, max_new_tokens=4096,
              temperature=0.0) -> VLMResponse:

    # 1) 首次调用
    raw, elapsed = self.call(...)
    data = extract_json(raw)       # 从任意文本中提取第一个合法 JSON dict

    # 2) 无 schema → 直接返回
    if schema is None:
        return VLMResponse(raw, data, elapsed, 1, True, [])

    # 3) Schema 校验
    ok, errs = validate_output(schema, data)
    if ok:
        return VLMResponse(raw, data, elapsed, 1, True, [])

    # 4) 一次结构化重试：把上一份 JSON + 精确错误信息喂回去
    retry_text = f"{user_text}\n上一次输出:\n{json.dumps(data)}\n错误: {errs}"
    raw2, e2 = self.call(system_prompt, images, retry_text, temperature=0.0)
    data2 = extract_json(raw2)
    ok2, errs2 = validate_output(schema, data2)
    return VLMResponse(raw2, data2, elapsed+e2, 2, ok2, errs2 if not ok2 else [])
```

**设计要点**: 重试时带上上一份 JSON 全文 + 精确错误，让 VLM 局部修正而非重新生成空壳。

### 4.4 JSON 提取 `extract_json()`

用括号计数法（处理字符串内的 `{}` 和转义字符），无需 VLM 输出带 markdown code block 也能正确解析。

---

## 5. Prompt 体系

**文件**: `pipeline/prompts.py`

### 5.1 workflow_planning_prompt() — Phase 1 系统提示词

```python
def workflow_planning_prompt() -> str:
    return (
        # 1. 角色设定
        "你是地球物理AI工作流规划器..."

        # 2. 可用模型清单（动态生成）
        + available_models_desc()

        # 3. 任务→模型推荐映射
        + TASK_MODEL_MAP
        # 断层 → seismic_domain_model, 层位 → horizon_tracker,
        # 沉积相 → facies_classifier, 裂缝 → seismic_domain_model

        # 4. 每个模型的硬性字段要求（与 schema allOf 一致）

        # 5. 4个 few-shot 示例
        #   - 地震断层检测 (seismic_domain_model + yolo_world)
        #   - 测井分析 (well_log_analyzer + traditional_code)
        #   - 沉积相分类 (attribute_extractor + facies_classifier)
        #   - 层位追踪 (horizon_tracker)
    )
```

### 5.2 VERIFICATION_PROMPT — Phase 3 验证系统提示词

包含 5 类地质判断准则：

| 检测类型 | ✓ 真 | ✗ 假阳性 |
|----------|------|----------|
| **断层** | 同相轴可见垂向错断 + 反射终止 | 仅振幅渐变无错断、河道边缘 |
| **层位** | 横向同相位连续追踪 | 穿相位、在断层处穿过 |
| **沉积相** | 同反射构型归一类，地质可解释 | 不同构型混类、振幅强弱驱动 |
| **含气检测** | 亮点 + 低频阴影 + 可能平点 | 仅振幅异常无低频响应 |
| **测井岩性** | GR低+交会一致 | 致密层(高DEN低GR)、有机质页岩 |

### 5.3 TASK_MODEL_MAP — 任务推荐

```python
TASK_MODEL_MAP = """任务→下游模型推荐:
  断层(fault):  首选 seismic_domain_model, 次选 yolo_world
  层位(horizon): 首选 horizon_tracker, 次选 sam
  沉积相(facies): 首选 facies_classifier, 次选 attribute_extractor+yolo_world
  裂缝(fracture): 首选 seismic_domain_model, 次选 attribute_extractor
  测井分析: well_log_analyzer, 传统阈值规则用 traditional_code
  储层预测: attribute_extractor → facies_classifier
  流体检测: well_log_analyzer → attribute_extractor"""
```

### 5.4 竞赛计划文本 `_build_competition_plan_text()`

赛题流程中，VLM 收到的 user text 是动态构建的：

```
=== 赛题任务 ===
sample_id: demo_sample_001
target_classes: ['fault', 'channel', 'reservoir_candidate']
  - fault: 推荐模型 seismic_domain_model
  - channel: 推荐模型 facies_classifier

=== 可用图像 (用 image_name 指定在哪个图上跑) ===
  1. image_name="seismic_inline"  view=inline  size=(1280, 960)
  2. image_name="seismic_crossline"  view=crossline  size=(1280, 960)
  3. image_name="seismic_slice"  view=time_or_depth_slice  size=(1280, 960)
  4. image_name="seismic_local_patch"  view=local_horizontal_patch
  5. image_name="well_log_panel"  view=well_log_panel  size=(1870, 1413)

=== 可用下游模型 ===
(available_models_desc() 全部 8 个模型)

=== 任务→模型推荐 ===
(TASK_MODEL_MAP)

请规划 workflow_steps...
```

---

## 6. 下游模型注册表

**文件**: `pipeline/downstream/base.py` `pipeline/downstream/__init__.py`

### 6.1 Protocol

```python
class DownstreamModel(Protocol):
    name: str              # 唯一标识符 (VLM通过这个引用)
    description: str       # 给VLM看的一句话能力描述
    required_fields: list[str]  # instruction 必需字段说明
    output_shape: str      # 输出格式描述
    def detect(self, instruction: dict, image=None, context=None) -> list[dict]:
        """执行检测。instruction 来自 VLM 的规划输出。"""
```

### 6.2 注册机制

```python
# base.py
_REGISTRY: dict[str, DownstreamModel] = {}

def register(model):  # 同名覆盖
def get(name) -> DownstreamModel | None

# __init__.py — import 时自动注册全部 8 个
def bootstrap_defaults():
    register(YoloWorld())
    register(Sam())
    register(TraditionalCode())
    register(HorizonTracker())
    register(FaciesClassifier())
    register(WellLogAnalyzer())
    register(AttributeExtractor())
    try:
        register(SeismicDomainDetector())  # 大依赖，可选
    except ImportError: pass

bootstrap_defaults()
```

### 6.3 VLM 感知机制

`available_models_desc()` 动态遍历注册表，生成模型清单文本。VLM 通过这个文本"认识"每个模型。新增模型只需三步：实现 Protocol → 注册 → 更新 Schema 枚举。

---

## 7. 8 个下游模型

### 7.1 yolo_world — 开放词汇目标检测

| 属性 | 内容 |
|------|------|
| 文件 | `yolo_world.py` |
| 状态 | ✅ 真实 (ultralytics YOLO-World) |
| 权重 | `weights/yolov8s-world.pt` |
| 输入 | `categories[{class_name, expected_range, confidence_threshold}]` |
| 输出 | `[{id, class_name, bbox_pixel, bbox_norm, confidence}]` |

**算法**: `YOLOWorld.set_classes(prompts)` → `model.predict(image)` → bbox 输出

**特点**: 
- 支持 `detect_open_vocab()` 逐 ROI 检测
- 加载失败自动 fallback mock
- 输出同时含 pixel 和 normalized bbox

### 7.2 seismic_domain_model — 相干+结构张量断层检测

| 属性 | 内容 |
|------|------|
| 文件 | `seismic_domain_model.py` |
| 状态 | ✅ 真实 (scipy + numpy) |
| 输入 | `{task, attribute(coherence|gradient|variance|structure_tensor|both), confidence_threshold, regions_of_interest, min_region_area_pixels}` |
| 输出 | `[{id, class_name, bbox_pixel, confidence, area_pixels, aspect_ratio}]` |

**算法**:
- **相干体** (`coherence_map`): trace-to-trace semblance 滑动窗口互相关
- **梯度断层概率** (`gradient_fault_prob`): 垂直梯度 gy − 0.5×水平梯度 gx → 高斯平滑
- **局部方差** (`local_variance`): 滑动窗口方差（AGC数据有效）
- **结构张量** (`structure_tensor_edge`): 2D 梯度结构张量特征值比

**ROI 模式**: VLM 圈 ROI → 领域模型在 ROI 内用×0.7 更低阈值精细检测 → NMS 去重

**形状评分**: 高度>>宽度（垂直延伸）→ 高置信断层；团块状 → 降权

### 7.3 horizon_tracker — 互相关层位追踪

| 属性 | 内容 |
|------|------|
| 文件 | `horizon_tracker.py` |
| 状态 | ✅ 真实 (numpy correlate) |
| 输入 | `{seed_points[{trace_idx, sample_idx}], tracking_mode(peak|trough|correlation|zero_crossing), horizon_name, search_window_samples}` |
| 输出 | `[{id, horizon_name, points[{trace_idx, sample_idx, confidence}], continuity_score, average_confidence}]` |

**算法**: 
1. 在种子道提取模板波形 (window_half=25 采样点)
2. Z-score 归一化模板
3. 对相邻道用 `np.correlate(trace, template, mode='valid')` 一次性计算全部时移的互相关
4. 取最大相关位置作为新追踪点
5. 双向传播，相关度低于 0.4 终止

**特点**: 
- `np.correlate` 比逐点滑动更鲁棒（不受窄波峰影响）
- 多种子点 → 多个独立追踪线段
- continuity_score: 连续追踪比例

### 7.4 facies_classifier — 多属性沉积相分类

| 属性 | 内容 |
|------|------|
| 文件 | `facies_classifier.py` |
| 状态 | ✅ 真实 (sklearn 或 scipy fallback) |
| 输入 | `{n_clusters(2-20), attribute_list[envelope,phase,...], method(kmeans|gmm), regions_of_interest}` |
| 输出 | `[{id, cluster_id, area_pixels, centroid_xy, cluster_center{attr_values}, dominant_feature}]` |

**算法**:
1. 多属性提取 (envelope/phase/frequency/gradient/local_variance/local_entropy/dip)
2. Z-score 归一化 → PCA 降维 (3维, 保留95%方差)
3. GMM (默认) 或 KMeans 聚类
4. 每簇计算: 面积、质心、属性中心、最具区分性属性

**fallback**: sklearn 不可用时用 `scipy.cluster.vq.kmeans2`

### 7.5 well_log_analyzer — 测井曲线分析

| 属性 | 内容 |
|------|------|
| 文件 | `well_log_analyzer.py` |
| 状态 | ✅ 真实 (scipy + 交会图规则) |
| 输入 | `{analysis_type(curve_segmentation|lithology_classification|fluid_identification|full_analysis), rules, depth_range}` |
| 输出 | `[{id, class_name, depth_top_m, depth_bottom_m, confidence, lithology?, fluid_type?, evidence[]}]` |

**算法**:
- **曲线分割**: 梯度 Z-score 变点检测 → 合并邻近峰值
- **岩性分类**: GR/DEN/RT 交会规则 (clean_sandstone < 45API → silty → shaly → shale > 90API)
- **流体识别**: RT/DEN/CNL 交会 (RT>50+DEN<2.30+CNL<0.18 → gas, RT>20+DEN<2.35 → oil, RT<5 → water)

**输入来源**: `context["curves"]` 含真实曲线数据 → 精确分析；无数据时用 VLM rules fallback

### 7.6 attribute_extractor — 多属性提取

| 属性 | 内容 |
|------|------|
| 文件 | `attribute_extractor.py` |
| 状态 | ✅ 真实 (scipy.signal) |
| 输入 | `{attributes[envelope,phase,frequency,sweetness,glcm,spectral,...], spectral_bands, regions_of_interest}` |
| 输出 | `[{id, attribute_name, statistics{min,max,mean,std}, roi_index}]` |

**属性类别**:
- **瞬时**: envelope, phase, frequency (`scipy.signal.hilbert`)
- **纹理**: GLCM energy/contrast/homogeneity/correlation (`skimage.feature.graycomatrix`)
- **频谱**: 多频带 Butterworth 带通滤波
- **复合**: RMS振幅, 甜点 (=envelope/√frequency)

### 7.7 sam — 分割 (待接入)

| 属性 | 内容 |
|------|------|
| 文件 | `sam.py` |
| 状态 | ⚠️ mock |

接入 SAM-2 替换 `detect()` 内部即可。VLM 无需感知变化。

### 7.8 traditional_code — 规则引擎 (待升级)

| 属性 | 内容 |
|------|------|
| 文件 | `traditional_code.py` |
| 状态 | ⚠️ mock（well_log_analyzer 已覆盖其测井功能） |

保留用于兼容旧的 VLM prompt 格式。`well_log_analyzer` 提供同等的 rules→depth_ranges 能力。

---

## 8. 输出与导出

**文件**: `pipeline/exporter.py`

### 8.1 输出产物

```
out/
├── vlm_plan.json           # Phase 1 VLM 规划结果
├── report.json             # 汇总报告
├── fault_attribute.sgy     # 每个 target_class 一个 3D 属性 SEG-Y
├── channel_attribute.sgy
├── reservoir_candidate_attribute.sgy
├── *_inline.png            # 标注 PNG (bbox 叠原图)
└── *_crossline.png
```

### 8.2 坐标反变换

```python
# exporter.py:177-267 aggregate_adapter_detections()
# bbox_norm [x1,y1,x2,y2] in [0,1] → 通过 manifest.views[view].source_indices
# → 在 3D cube 的对应位置涂入 confidence
# 支持 inline/crossline/slice 三种 view
```

### 8.3 导出函数

| 函数 | 输入 | 输出 |
|------|------|------|
| `export_json(result, geom, task, dir)` | AgentResult | JSON |
| `export_annotated_png(result, image, geom, task, dir)` | AgentResult+PIL | bbox标注PNG |
| `build_slice_mask(result, geom, shape, task)` | AgentResult | (H,W) numpy mask |
| `export_volume_attribute(vol, masks, task, dir)` | 逐切片mask | 3D SEG-Y |
| `aggregate_adapter_detections(dets, manifest, shape)` | YOLO检测 | {class: 3D cube} |
| `summary_report(results, dir)` | dict | report.json |

---

## 9. IO 模块

### 9.1 SEG-Y (`io/segy.py`)

```python
@dataclass
class SegyVolume:
    cube: np.ndarray          # (n_il, n_xl, n_samples) float32
    inlines: np.ndarray
    xlines: np.ndarray
    sample_interval_ms: float
    n_samples: int
```

| 函数 | 功能 |
|------|------|
| `read_segy(path)` | 读取 SEG-Y → SegyVolume |
| `write_attribute_segy(vol, attr, path)` | 写属性 SEG-Y（复制头或从零构造） |
| `extract_inline_slice(vol, idx)` | 提取 inline 切片 (n_samples, n_xl) |
| `extract_xline_slice(vol, idx)` | 提取 xline 切片 (n_samples, n_il) |
| `extract_time_slice(vol, idx)` | 提取时间切片 (n_il, n_xl) |
| `synthetic_volume(...)` | 合成测试 SEG-Y |

### 9.2 坐标几何 (`io/geometry.py`)

```python
@dataclass
class SliceGeometry:
    axis_x_name: str     # "CDP" | "inline" | "crossline"
    axis_y_name: str     # "time_ms" | "depth_m"
    x_min, x_max: float
    y_top, y_bottom: float
    pixel_width, pixel_height: int
```

`pixel_to_data(bbox_pixel, geom)` → 数据坐标 dict
`data_to_pixel(x, y, geom)` → 像素坐标

### 9.3 切片渲染 (`io/render.py`)

`render_slice(array2d, ...)` → `(PIL.Image, SliceGeometry)`

用 matplotlib 渲染 2D 地震数组为 PIL 图像，同时输出几何元数据。

---

## 10. Schema 体系

**文件**: `schemas/output_schemas.py`

### 10.1 8 套 Schema

| Schema | 用途 |
|--------|------|
| `SEISMIC_OUTPUT_SCHEMA` | Agent 1 地震解释 (faults/horizons/facies/anomalies) |
| `LOG_OUTPUT_SCHEMA` | Agent 2 测井分析 (lithology/reservoir/fluid zones) |
| `FUSION_OUTPUT_SCHEMA` | Agent 3 井震融合 (calibration/interfaces/correlation) |
| `PROSPECT_OUTPUT_SCHEMA` | Agent 4 目标评价 (targets/risks/decisions) |
| `WORKFLOW_PLAN_SCHEMA` | **核心**: VLM 工作流规划输出 |
| `WORKFLOW_VERIFICATION_SCHEMA` | VLM 验证输出 |

### 10.2 WORKFLOW_PLAN_SCHEMA 结构

```json
{
  "required": ["scene_understanding", "workflow_steps"],
  "properties": {
    "scene_understanding": {"type": "string", "minLength": 1},
    "workflow_steps": {
      "type": "array", "minItems": 1,
      "items": {
        "required": ["step", "model", "instruction"],
        "properties": {
          "step": {"type": "integer", "minimum": 1},
          "model": {"enum": ["yolo_world", "sam", "traditional_code",
                     "seismic_domain_model", "attribute_extractor",
                     "horizon_tracker", "facies_classifier",
                     "well_log_analyzer"]},
          "image_name": {"type": "string"},
          "reason": {"type": "string"},
          "instruction": {"type": "object"}
        },
        "allOf": [
          // 每个 model 的 if-then 约束
          // 如 model="horizon_tracker" → instruction 必须含 seed_points+tracking_mode
          // 如 model="seismic_domain_model" → instruction 必须含 task
          // ... (8个模型的约束)
        ]
      }
    }
  }
}
```

### 10.3 validate_output()

```python
def validate_output(schema, data) -> tuple[bool, list[str]]:
    import jsonschema
    try:
        jsonschema.validate(data, schema)
        return True, []
    except jsonschema.ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        return False, [f"{path}: {e.message}"]
```

错误信息直接作为 VLM 重试的反馈。

---

## 11. CLI 使用

### 环境变量

| 变量 | 默认值 |
|------|--------|
| `QWEN_VL_PATH` | `/data/.../Qwen3-VL-8B-Instruct` |
| `YOLO_WORLD_PATH` | `weights/yolov8s-world.pt` |

### 赛题主流程

```bash
# 1) geo_adapter 前置
geo-adapter prepare --config sample.yaml
# → runs/demo_sample_001/

# 2) 本 pipeline
CUDA_VISIBLE_DEVICES=1 python -m pipeline \
    --run-dir runs/demo_sample_001 \
    --output-dir out/ \
    --max-iter 3

# 关闭验证回环（快一倍）
python -m pipeline --run-dir runs/demo --no-verify
```

### Fallback SEG-Y

```bash
python -m pipeline --input volume.sgy \
    --tasks fault,horizon --slice-axis inline \
    --output-dir out/
```

### 4-Agent 语义解释

```bash
python -m pipeline --agent all \
    --seismic-image section.png --log-image log.png \
    --output out.json
```

---

## 12. 扩展指南

### 新增下游模型

1. **创建**: `pipeline/downstream/my_model.py`

```python
class MyModel:
    name = "my_model"
    description = "一句话描述给VLM看"
    required_fields = ["param1", "param2?"]
    output_shape = "list[{id, result}]"

    def detect(self, instruction, image=None, context=None) -> list[dict]:
        return [{"id": "m1", "result": ...}]
```

2. **注册**: 在 `__init__.py` 的 `bootstrap_defaults()` 中 `register(MyModel())`

3. **Schema**: 在 `output_schemas.py` WORKFLOW_PLAN_SCHEMA 的 `model` enum 添加 `"my_model"`，可选添加 `allOf` 约束

4. **Prompt**: 在 `prompts.py` 的 `workflow_planning_prompt()` 中添加字段说明，在 `TASK_MODEL_MAP` 中添加推荐场景

### 新增地质任务

在 `tasks.py`:

```python
TASKS["new_task"] = GeologicalTask(
    name="new_task",
    yolo_classes=["class1", "class2"],
    description="中文地质描述",
    downstream_hint="backup_model",
    recommended_model="primary_model",
    overlay_color="#00ff00",
)
CLASS_ALIASES["别名"] = "new_task"
```

---

## 13. 文件索引

```
oil-gas-llm/
├── pipeline/
│   ├── __init__.py              # 公共 API 导出
│   ├── __main__.py              # CLI 入口 (3种模式)
│   ├── vlm.py                   # VLMClient + JSON提取 + Schema重试
│   ├── prompts.py               # 集中管理全部 Prompt 模板
│   ├── agents.py                # LoopAgent(闭环) + SingleShotAgent
│   ├── orchestrator.py          # Pipeline 统一编排 (3个入口)
│   ├── adapter.py               # geo_adapter RunPackage 载入
│   ├── tasks.py                 # 地质任务注册表 + CLASS_ALIASES
│   ├── exporter.py              # AgentResult → PNG/JSON/SEG-Y
│   ├── downstream/
│   │   ├── base.py              # DownstreamModel Protocol + get/register
│   │   ├── __init__.py          # bootstrap_defaults() 注册全部模型
│   │   ├── yolo_world.py        # YOLO-World (✅真实)
│   │   ├── seismic_domain_model.py  # 相干+结构张量 (✅真实)
│   │   ├── horizon_tracker.py   # 互相关层位追踪 (✅真实)
│   │   ├── facies_classifier.py # PCA+GMM沉积相 (✅真实)
│   │   ├── well_log_analyzer.py # 曲线分割+岩性+流体 (✅真实)
│   │   ├── attribute_extractor.py   # 多属性提取 (✅真实)
│   │   ├── sam.py               # SAM (mock)
│   │   └── traditional_code.py  # 规则引擎 (mock)
│   └── io/
│       ├── segy.py              # SEG-Y 读写
│       ├── geometry.py          # SliceGeometry + pixel↔data
│       └── render.py            # 数组 → PIL 图像
├── schemas/
│   └── output_schemas.py        # 6套JSON Schema + validate_output
├── prompts/                     # Agent System Prompt markdown 文件
├── weights/yolov8s-world.pt     # YOLO-World checkpoint
├── test/                        # 71 个单元测试
│   ├── test_pipeline_unit.py    # 19: extract_json/schema/registry
│   ├── test_io_unit.py          # 12: SEG-Y/几何/exporter
│   ├── test_adapter_unit.py     # 14: adapter/tasks/aggregate
│   └── test_downstream_unit.py  # 26: 8个模型实例化+输出格式+schema
├── docs/
│   └── PIPELINE_TECHNICAL_DOC.md  # 本文档
└── requirements.txt             # 依赖声明
```
