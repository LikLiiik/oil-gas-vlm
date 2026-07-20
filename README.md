# Oil-Gas VLM — 油气地球物理多模态Agent工作流

基于本地部署 VLM (Qwen3-VL-8B) 实现地震图像与测井曲线的多模态特征融合，智能识别有利油气地质目标。

## 架构

```
┌──────────────────────────────────────────────────────────┐
│                    两阶段混合架构                          │
├──────────────────────────────────────────────────────────┤
│  Stage 1: Qwen3-VL-8B (10-17s)                          │
│  → 看图像 → 识别岩性/构造/流体类型 + 大致区间              │
│                                                          │
│  Stage 2: 传统岩石物理代码 (<0.01s)                       │
│  → GR阈值精确定位砂岩边界 (±0.1m)                         │
│  → RT+DEN阈值精确判断流体类型                             │
├──────────────────────────────────────────────────────────┤
│  结果: VLM语义理解 + 代码数值精度 = 100%准确率             │
└──────────────────────────────────────────────────────────┘
```

## 目录结构

```
oil-gas-vlm/
├── prompts/                              # 4个Agent的Few-shot Prompt模板
│   ├── seismic_interp_agent.md           # 地震剖面解释
│   ├── log_analysis_agent.md             # 测井曲线分析
│   ├── well_seismic_fusion_agent.md      # 井震多模态融合
│   └── prospect_evaluation_agent.md      # 有利目标综合评价
├── schemas/
│   └── output_schemas.py                 # JSON Schema定义 + 校验工具
├── pipeline/
│   ├── data_processing.md                # 数据处理流程说明
│   ├── iteration_notes.md                # Prompt迭代记录
│   └── accuracy_report.md                # 准确性评测报告
├── test/
│   ├── test_live.py                      # 全量功能测试
│   ├── test_accuracy.py                  # 准确性评测（含ground truth）
│   └── test_two_stage.py                 # 两阶段策略验证
├── .gitignore
└── README.md
```

## 快速开始

### 环境

```bash
conda activate qwen35grpo  # 或任何有 torch>=2.7, transformers>=5.0 的环境
pip install matplotlib Pillow
```

### 部署模型

```bash
# Qwen3-VL-8B (推荐，非thinking，10s推理)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir /data/models/qwen3-vl-8b
```

### 测试

```bash
# 单个Agent测试
CUDA_VISIBLE_DEVICES=1 python test/test_live.py

# 准确性评测
CUDA_VISIBLE_DEVICES=1 python test/test_accuracy.py

# 两阶段策略验证
CUDA_VISIBLE_DEVICES=1 python test/test_two_stage.py
```

## 四个Agent

| Agent | 输入 | 输出 | 模型 |
|-------|------|------|------|
| **SeismicInterpAgent** | 地震剖面图(PNG) | 断层、层位、地震相 JSON | Qwen3-VL-8B |
| **LogAnalysisAgent** | 测井曲线图(PNG) → 两阶段代码精确定位 | 岩性、物性、流体 JSON | VLM粗分+代码精确 |
| **WellSeismicFusionAgent** | 井震对比图(PNG) | 井震标定、关联分析 JSON | Qwen3-VL-8B |
| **ProspectEvaluationAgent** | 前序JSON文本 | 目标排序、风险评估 JSON | Qwen3-VL-8B |

## 关键参数

```python
model.generate(
    max_new_tokens=32768,    # 大token上限确保JSON完整
    do_sample=True,
    temperature=0.3,
    repetition_penalty=1.1,
    top_p=0.95,
)
```

## 评测结果

| 指标 | Qwen3.5-9B(thinking) | Qwen3-VL-8B + 两阶段 |
|------|---------------------|----------------------|
| Think浪费 | 57-81% | 0% |
| 推理时间 | 141-164s | 10s |
| JSON完整率 | 30% | 100% |
| Sand召回 | 50% | 100% |
| Fluid准确率 | 33% | 100% |
| 深度精度 | ±50-200m | ±0.1m |
