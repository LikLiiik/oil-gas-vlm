"""集中管理 VLM prompt 模板。"""
from __future__ import annotations

from pathlib import Path

from .downstream import available_models_desc

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_system_prompt(md_file: str) -> str:
    """从 prompts/*.md 文件里读 `## System Prompt` 之后到下一个 `##` 之前的内容。"""
    path = PROMPTS_DIR / md_file
    text = path.read_text(encoding="utf-8")
    marker = "## System Prompt"
    idx = text.find(marker)
    if idx < 0:
        return text
    body = text[idx + len(marker):]
    next_h2 = body.find("\n## ")
    if next_h2 > 0:
        body = body[:next_h2]
    return body.strip()


# ---- Task → Model 推荐映射表（供 VLM 参考）--------------------------------

TASK_MODEL_MAP = """任务→下游模型推荐:
  - 断层(fault):  首选 cig_fault（HRNet预训练权重,3D断层分割），
                  次选 seismic_domain_model（相干/梯度属性检测）
  - 层位(horizon): 首选 horizon_tracker（相位自动追踪），
                   次选 sam（轻量分割连续反射）
  - 沉积相(facies): 首选 facies_classifier（多属性聚类分相），
                    次选 seismic_foundation + attribute_extractor（SFM特征+纹理）
  - 裂缝(fracture): 首选 seismic_domain_model（相干/方差属性），
                    次选 attribute_extractor（多属性异常检测）
  - 测井分析: well_log_ml（RF岩性分类+流体识别, 岩石物理知识训练），
            well_log_analyzer（changepoint分割/交会规则），
            传统阈值规则用 traditional_code
  - 储层预测: attribute_extractor（甜点/频谱/RMS）→ facies_classifier
  - 流体检测: well_log_analyzer（RT/DEN交会）
  - 地震相特征: seismic_foundation（SFM seismic预训练ViT特征提取）"""


# ---- Autonomous workflow (test_loop.py 用) --------------------------------

_PLANNING_FEWSHOT_SEISMIC = '''\
示例1（地震剖面—断层检测场景）:
{
  "scene_understanding": "地震剖面显示 CDP 60-100 存在同相轴错断（疑似断层），CDP 160-220 / 1100-1400ms 存在强负振幅（疑似亮点）",
  "workflow_steps": [
    {
      "step": 1,
      "model": "seismic_domain_model",
      "reason": "断层检测需要相干/梯度属性，domain model 专于此任务",
      "instruction": {
        "task": "fault_detection",
        "attribute": "gradient",
        "confidence_threshold": 0.3,
        "min_region_area_pixels": 80,
        "regions_of_interest": [
          {"bbox_xyxy_norm": [0.15, 0.3, 0.35, 0.65],
           "desc": "CDP 60-100 区域疑似断层"}
        ]
      }
    },
    {
      "step": 2,
      "model": "seismic_foundation",
      "reason": "SFM seismic预训练ViT提取特征，辅助验证断层区域的异常模式",
      "instruction": {
        "task": "feature_extraction",
        "regions_of_interest": [
          {"bbox_xyxy_norm": [0.4, 0.5, 0.7, 0.75],
           "desc": "CDP 160-220 亮点异常区"}
        ]
      }
    }
  ],
  "verification_strategy": "per_step",
  "max_iterations": 2
}'''

_PLANNING_FEWSHOT_LOG = '''\
示例2（测井曲线场景）:
{
  "scene_understanding": "GR 曲线在 1200-1255m、1550-1625m 处显著低于 50 API（砂岩段）；RT 在 1555-1595m 高阻异常，DEN 同段偏低（含气特征）",
  "workflow_steps": [
    {
      "step": 1,
      "model": "well_log_analyzer",
      "reason": "需要精确曲线分割+岩性分类+流体识别，well_log_analyzer 一步完成",
      "instruction": {
        "analysis_type": "full_analysis",
        "depth_range": {"top_m": 1100, "bottom_m": 1900}
      }
    },
    {
      "step": 2,
      "model": "traditional_code",
      "reason": "对特定规则（GR<50）做精确数值提取，补充第一步的分析",
      "instruction": {
        "rules": [
          {"class_name": "low_GR_sandstone",
           "rule": "GR < 50",
           "expected_depth_ranges": [
             {"top_m": 1200, "bottom_m": 1255},
             {"top_m": 1550, "bottom_m": 1625}
           ]},
          {"class_name": "high_resistivity_pay",
           "rule": "RT > 20",
           "expected_depth_ranges": [{"top_m": 1555, "bottom_m": 1595}]}
        ]
      }
    }
  ],
  "verification_strategy": "batch",
  "max_iterations": 2
}'''

_PLANNING_FEWSHOT_FACIES = '''\
示例3（沉积相分析场景）:
{
  "scene_understanding": "剖面显示多种反射构型：上部平行连续（陆棚），中部S形前积（三角洲），下部杂乱+透镜状（浊积/河道）",
  "workflow_steps": [
    {
      "step": 1,
      "model": "attribute_extractor",
      "reason": "先提取多种属性供后续聚类使用",
      "instruction": {
        "attributes": ["envelope", "frequency", "sweetness", "rms_amplitude"]
      }
    },
    {
      "step": 2,
      "model": "facies_classifier",
      "reason": "多属性 PCA+GMM 聚类划分相带，VLM 后续地质标定",
      "instruction": {
        "n_clusters": 5,
        "attribute_list": ["envelope", "frequency", "sweetness"],
        "method": "gmm"
      }
    }
  ],
  "verification_strategy": "batch",
  "max_iterations": 1
}'''

_PLANNING_FEWSHOT_HORIZON = '''\
示例4（层位追踪场景）:
{
  "scene_understanding": "CDP 50-250 存在一条横向连续强反射轴 T3，波峰特征明显，约在 1200-1300ms 之间缓慢倾斜",
  "workflow_steps": [
    {
      "step": 1,
      "model": "horizon_tracker",
      "reason": "连续强反射轴适合相位互相关自动追踪",
      "instruction": {
        "seed_points": [
          {"trace_idx": 80, "sample_idx": 300},
          {"trace_idx": 180, "sample_idx": 312}
        ],
        "tracking_mode": "correlation",
        "search_window_samples": 15,
        "horizon_name": "T3_mfs"
      }
    }
  ],
  "verification_strategy": "per_step",
  "max_iterations": 2
}'''


def workflow_planning_prompt() -> str:
    return (
        "你是地球物理AI工作流规划器。分析图像后，自主决定需要调用哪些下游模型、"
        "按什么顺序、用哪些参数。\n\n"
        f"{available_models_desc()}\n\n"
        "=== 任务→模型推荐（请严格遵循，不要给任务错配模型）===\n"
        f"{TASK_MODEL_MAP}\n\n"
        "=== 硬性字段要求（不满足会被 schema 拒绝）===\n"
        "- traditional_code.instruction 必须包含 rules 数组，每项至少含 class_name 和 rule；"
        "expected_depth_ranges 用 [{top_m, bottom_m}] 形式（数值，不要字符串）。\n"
        "- seismic_domain_model.instruction 含 task(fault_detection)、attribute(gradient|coherence|structure_tensor|variance)、"
        "confidence_threshold、regions_of_interest[{bbox_xyxy_norm:[x1,y1,x2,y2],desc}]。\n"
        "- sam.instruction 必须包含 prompt_type(point|bbox)、prompt_value、label。\n"
        "- horizon_tracker.instruction 必须包含 seed_points 数组([{trace_idx, sample_idx},...])、"
        "tracking_mode(peak|trough|correlation|zero_crossing)；可选 search_window_samples、horizon_name。\n"
        "- facies_classifier.instruction 必须包含 n_clusters(2-20)；"
        "可选 attribute_list([envelope,phase,frequency,...])、regions_of_interest、method(kmeans|gmm)。\n"
        "- well_log_analyzer.instruction 必须包含 analysis_type("
        "curve_segmentation|lithology_classification|fluid_identification|full_analysis)；"
        "可选 rules[]、depth_range{top_m, bottom_m}。\n"
        "- attribute_extractor.instruction 必须包含 attributes 数组(如['envelope','sweetness','spectral_20hz'])；"
        "可选 regions_of_interest[]、spectral_bands。\n"
        "- seismic_foundation.instruction 必须包含 task(facies_classification|feature_extraction)。\n"
        "- cig_fault.instruction 可选 threshold(0-1, 默认0.5)、scale(默认1.0)。\n"
        "HRNet预训练断层检测，权重自动从ModelScope下载(~40MB)。\n"
        "- cig_channel.instruction 可选 threshold、scales(多尺度列表)。\n"
        "HRNet预训练河道检测，多尺度集成预测。\n"
        "- well_log_ml.instruction 必须包含 analysis_type(lithology|fluid|full)。\n"
        "RandomForest岩石物理知识模型, 6类岩性+流体识别, 模型缓存到~/.cache。\n\n"
        # few-shot 示例
        f"{_PLANNING_FEWSHOT_SEISMIC}\n\n"
        f"{_PLANNING_FEWSHOT_LOG}\n\n"
        f"{_PLANNING_FEWSHOT_FACIES}\n\n"
        f"{_PLANNING_FEWSHOT_HORIZON}\n\n"
        "现在请按同样格式，针对给定图像输出一个工作流计划。仅输出JSON。"
    )


VERIFICATION_PROMPT = """你是地球物理验证专家。根据原始图像验证下游模型的每条检测结果。

=== 地质真伪判断准则 ===

断层（fault）验证标准:
  ✓ 真断层: 同相轴可见垂向/倾斜错断 + 反射终止 + 可能伴生牵引构造/断面波
  ✗ 假阳性: 仅振幅渐变但无错断、河道边缘反射终止(单侧)、处理噪声条带、采集脚印
  断距评估: 小(<10ms)、中(10-30ms)、大(>30ms)

层位（horizon）验证标准:
  ✓ 真层位: 横向上同相位连续追踪、振幅/波形稳定或渐变、代表可对比的地层界面
  ✗ 假阳性: 穿相位(追踪到相邻相位)、在断层处穿过、信噪比极低区域强制连续

沉积相（facies）验证标准:
  ✓ 合理分类: 同反射构型归为一类(平行→平行, 前积→前积)、地质上可解释
  ✗ 不合理: 将不同构型混为一类、明显受振幅强弱而非构型驱动

含气检测（gas detection）验证标准:
  ✓ 证据充分: 强振幅异常(亮点) + 低频阴影 + 可能伴生平点
  ✗ 证据不足: 仅振幅异常无低频响应、或异常在上倾方向不封堵

测井岩性验证标准:
  ✓ 合理: GR低(<60)对应砂岩/碳酸盐岩、GR高(>90)对应泥岩、与RT/DEN交会一致
  ✗ 不合理: GR低但DEN高(致密层)、GR高但RT极低(可能为有机质页岩)

=== 工作流 ===

原始工作流: {workflow_plan}
下游执行结果: {downstream_results}

逐条验证:
1. 这条检测是否真实地质特征？（对照原始图像判断，使用上述准则）
2. 如果是假阳性，原因是什么？（参考上述✗ 类别）
3. 有没有遗漏的目标（图像可见但模型没检测到）？
4. 是否需要追加步骤或调整参数重新检测？

JSON:
{"verified":[{"step":1,"model":"seismic_domain_model","result_id":"...","is_real":true/false,
   "confidence":0.9,"geological_reason":"...","rejection_reason":null}],
 "false_positives":2,
 "missed_targets":[{"suggested_model":"...","class_name":"...","expected_range":{...},"reason":"..."}],
 "need_retry":true/false,
 "retry_instructions":{"step":1,"model":"...","adjusted_params":{...},"reason":"..."},
 "final_summary":"验证总结"}"""


# ---- 四个独立 Agent 的 system prompt（从 prompts/*.md 读取） --------------

def seismic_interp_prompt() -> str:
    return _load_system_prompt("seismic_interp_agent.md")


def log_analysis_prompt() -> str:
    return _load_system_prompt("log_analysis_agent.md")


def well_seismic_fusion_prompt() -> str:
    return _load_system_prompt("well_seismic_fusion_agent.md")


def prospect_evaluation_prompt() -> str:
    return _load_system_prompt("prospect_evaluation_agent.md")
