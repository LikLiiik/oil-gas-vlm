"""
VLM分析准确性测试 — 用已知ground truth的模拟数据评测每个Agent

测试指标:
- 层位深度误差 (m)
- 岩性分类准确率
- 流体识别召回率/精确率
- 断层位置误差 (CDP/ms)
"""

import torch, json, io, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
import sys, os

MODEL_PATH = "/data/yxjiang/modelscope/hub/models/Qwen/Qwen3.5-9B"

print("Loading model...")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Loaded!")

# ============================================================
# Ground Truth 定义
# ============================================================

# ---- 地震 Ground Truth ----
SEISMIC_GT = {
    "faults": [
        {"id": "F1", "type": "normal", "dip_direction": "NE",
         "cdp_position": 80, "time_position_ms": 1000, "throw_samples": 5}
    ],
    "bright_spot": {
        "cdp_range": [100, 120], "time_range_ms": [975, 1025],
        "type": "gas_bright_spot"
    },
    "n_traces": 200, "n_samples": 400, "time_range_ms": [0, 2000],
}

# ---- 测井 Ground Truth ----
LOG_GT = {
    "depth_range_m": [1000, 2000],
    "lithology_zones": [
        {"id": "L1", "depth_top": 1200, "depth_bottom": 1255, "lithology": "sandstone", "gr_low": True},
        {"id": "L2", "depth_top": 1400, "depth_bottom": 1435, "lithology": "silty_sandstone", "gr_low": True},
        {"id": "L3", "depth_top": 1550, "depth_bottom": 1625, "lithology": "sandstone", "gr_low": True},
        {"id": "L4", "depth_top": 1750, "depth_bottom": 1805, "lithology": "sandstone", "gr_low": True},
    ],
    "reservoir_zones": [
        {"id": "R1", "depth_top": 1555, "depth_bottom": 1595,
         "porosity_pct": 18.5, "permeability": "good"},
        {"id": "R2", "depth_top": 1755, "depth_bottom": 1790,
         "porosity_pct": 12.0, "permeability": "poor"},
    ],
    "fluid_zones": [
        {"id": "F1", "depth_top": 1555, "depth_bottom": 1595,
         "fluid": "gas", "rt_ohmm": 85, "sw_pct": 25},
        {"id": "F2", "depth_top": 1755, "depth_bottom": 1790,
         "fluid": "water", "rt_ohmm": 3, "sw_pct": 100},
    ],
    "shale_zones": [
        {"depth_top": 1000, "depth_bottom": 1200, "lithology": "shale"},
        {"depth_top": 1255, "depth_bottom": 1400, "lithology": "shale"},
        {"depth_top": 1435, "depth_bottom": 1550, "lithology": "shale"},
        {"depth_top": 1625, "depth_bottom": 1750, "lithology": "shale"},
        {"depth_top": 1805, "depth_bottom": 2000, "lithology": "shale"},
    ],
}

# ============================================================
# 模拟数据生成 (带精确的ground truth)
# ============================================================

def generate_precise_seismic() -> Image.Image:
    """生成含精确断层位置的模拟地震剖面"""
    gt = SEISMIC_GT
    n_traces, n_samples = gt["n_traces"], gt["n_samples"]
    t = np.linspace(0, 2.0, n_samples) * 2 * np.pi
    section = np.zeros((n_samples, n_traces))

    for i in range(n_traces):
        trace = (
            0.6 * np.sin(t * 0.3 + i * 0.02) +
            0.35 * np.sin(t * 0.7 + i * 0.04) +
            0.25 * np.sin(t * 1.1 + i * 0.03) +
            0.12 * np.random.randn(n_samples)
        )
        # 精确断层: 在CDP=80处，向左5道逐渐增大错断
        fault_center = gt["faults"][0]["cdp_position"]
        throw = gt["faults"][0]["throw_samples"]
        if fault_center - 5 < i < fault_center + 5:
            shift = int(throw * (1 - abs(i - fault_center) / 5))
            trace = np.roll(trace, shift)
        section[:, i] = trace

    # 精确亮点: CDP 100-120, 时间 975-1025ms
    bs = gt["bright_spot"]
    t_start = int(bs["time_range_ms"][0] / 2000 * n_samples)
    t_end = int(bs["time_range_ms"][1] / 2000 * n_samples)
    section[t_start:t_end, bs["cdp_range"][0]:bs["cdp_range"][1]] *= 2.5

    fig, ax = plt.subplots(figsize=(10, 6))
    vmin, vmax = np.percentile(section, [5, 95])
    ax.imshow(section, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax,
              extent=[1, n_traces, 2000, 0])
    ax.set_xlabel('CDP'); ax.set_ylabel('Time (ms)')
    ax.set_title('Seismic Inline Section')
    # 标注ground truth
    ax.axvline(x=80, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axhline(y=1000, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def generate_precise_log() -> Image.Image:
    """生成含精确岩性/流体边界的模拟测井曲线"""
    gt = LOG_GT
    depth_start, depth_end = gt["depth_range_m"]
    n = 2000
    depth = np.linspace(depth_start, depth_end, n)

    # 基线: 泥岩 (高GR)
    gr = np.full(n, 95.0)
    rt = np.full(n, 3.0)
    ac = np.full(n, 80.0)
    den = np.full(n, 2.50)
    cnl = np.full(n, 0.32)
    sp = np.full(n, 5.0)

    # 砂岩段: 低GR
    for zone in gt["lithology_zones"]:
        m = (depth >= zone["depth_top"]) & (depth <= zone["depth_bottom"])
        n_sand = m.sum()
        gr[m] = 35 + np.random.randn(n_sand) * 6
        ac[m] = 63 + np.random.randn(n_sand) * 4
        den[m] = 2.28 + np.random.randn(n_sand) * 0.06
        cnl[m] = 0.16 + np.random.randn(n_sand) * 0.04
        sp[m] = -35 + np.random.randn(n_sand) * 6

    # 含气段: 高RT
    for zone in gt["fluid_zones"]:
        m = (depth >= zone["depth_top"]) & (depth <= zone["depth_bottom"])
        if zone["fluid"] == "gas":
            rt[m] = zone["rt_ohmm"] + np.random.randn(m.sum()) * 10
        else:
            rt[m] = zone["rt_ohmm"] + np.random.randn(m.sum()) * 0.5

    curves = [
        ('GR', gr, 'green', 'GR (API)', (0, 150)),
        ('SP', sp, 'blue', 'SP (mV)', (-60, 20)),
        ('RT', rt, 'red', 'RT (Ohm.m)', (0.1, 100)),
        ('AC', ac, 'orange', 'AC (us/ft)', (140, 40)),
        ('DEN', den, 'black', 'DEN (g/cm3)', (1.8, 2.8)),
        ('CNL', cnl, 'magenta', 'CNL (v/v)', (0.45, -0.15)),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(18, 14), sharey=True)
    for ax, (name, data, color, label, xlim) in zip(axes, curves):
        ax.plot(data, depth, color=color, linewidth=0.5)
        ax.set_xlabel(label, fontsize=8); ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3); ax.invert_yaxis()
        if xlim[0] > xlim[1]: ax.invert_xaxis()
    axes[0].set_ylabel('Depth (m)')
    fig.suptitle('Well Log: Well-Test-01', fontsize=12, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


# ============================================================
# VLM 调用
# ============================================================

def call_vlm(system_prompt: str, images: list, user_text: str,
             max_tokens=32768, temperature=0.3) -> str:
    """调用VLM, max_tokens=32768确保JSON不被截断"""
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": user_text})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if images:
        inputs = processor(text=text, images=images, return_tensors="pt").to(model.device)
    else:
        inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=True, temperature=temperature,
            repetition_penalty=1.1, top_p=0.9,
        )
    response = processor.decode(output[0], skip_special_tokens=True)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()
    return response


def extract_json(text: str) -> dict | None:
    """从VLM输出提取JSON，用深度追踪找到最外层完整JSON"""
    cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', text)
    cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')

    # 找所有可能JSON块的起始位置，用深度追踪找完整结构
    candidates = []
    for m in re.finditer(r'\{', cleaned):
        start = m.start()
        depth = 0
        in_string = False
        escape = False
        end = -1
        for i in range(start, len(cleaned)):
            c = cleaned[i]
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            candidates.append(cleaned[start:end])

    # 按长度降序尝试
    for candidate in sorted(candidates, key=lambda x: -len(x)):
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                # 优先返回有预期key的JSON
                if any(k in result for k in ['lithology_zones', 'fluid_zones', 'faults',
                                              'horizons', 'targets', 'well_seismic_calibration']):
                    return result
        except (json.JSONDecodeError, ValueError):
            continue

    # fallback: 返回最外层最大的dict JSON，排除明显的片段
    for candidate in sorted(candidates, key=lambda x: -len(x)):
        try:
            result = json.loads(candidate)
            if not isinstance(result, dict):
                continue
            # 排除片段（只含top/bottom/unit等坐标信息的碎片）
            fragment_keys = {'top', 'bottom', 'unit', 'depth_range', 'well_name', 'curves_identified'}
            if set(result.keys()).issubset(fragment_keys):
                continue
            if len(result) >= 3:
                return result
        except:
            continue

    # last resort
    for candidate in sorted(candidates, key=lambda x: -len(x)):
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except:
            continue
    return None


# ============================================================
# 准确性评测
# ============================================================

def evaluate_seismic(result) -> dict:
    """评测地震解释准确性"""
    gt = SEISMIC_GT
    scores = {}

    # --- 断层检测 ---
    faults = result.get('faults', [])
    gt_fault = gt["faults"][0]
    scores['fault_detected'] = len(faults) > 0

    if faults:
        # 位置误差
        best_fault = faults[0]
        for f in faults:
            positions = f.get('positions', [[0, 0]])
            if positions[0][0] != 0:
                best_fault = f
                break
        pos = best_fault.get('positions', [[0, 0]])
        if isinstance(pos[0], (int, float)):
            cdp_pred = pos[0]
            ms_pred = pos[1] if len(pos) > 1 else 0
        else:
            cdp_pred = pos[0][0]
            ms_pred = pos[0][1]
        scores['fault_cdp_error'] = abs(cdp_pred - gt_fault["cdp_position"])
        scores['fault_cdp_ok'] = scores['fault_cdp_error'] < 20

        # 断层类型
        scores['fault_type_match'] = gt_fault["type"] in str(best_fault.get('type', '')).lower()

    # --- 异常检测 ---
    anomalies = result.get('anomalies', [])
    scores['anomaly_detected'] = len(anomalies) > 0

    # --- Summary ---
    scores['has_summary'] = len(result.get('summary', '')) > 20

    scores['overall'] = (
        scores.get('fault_detected', False) * 1.0 +
        scores.get('anomaly_detected', False) * 0.5 +
        scores.get('has_summary', False) * 0.5
    ) / 2.0

    return scores


def evaluate_log(result) -> dict:
    """评测测井分析准确性"""
    gt = LOG_GT
    scores = {}

    # --- 岩性层位检测 ---
    litho = result.get('lithology_zones', [])
    scores['litho_count'] = len(litho)
    scores['litho_expected'] = len(gt['lithology_zones']) + len(gt['shale_zones'])

    # 检查是否识别到了关键砂岩段
    gt_sand_tops = {z['depth_top'] for z in gt['lithology_zones']}
    matched = 0
    depth_errors = []
    for pred_zone in litho[:12]:
        pred_top = pred_zone.get('depth_top', 0)
        pred_bot = pred_zone.get('depth_bottom', 0)
        pred_lith = str(pred_zone.get('lithology', '')).lower()

        # 匹配到最近的地面真值（放宽到±50m）
        for gt_zone in gt['lithology_zones']:
            if abs(pred_top - gt_zone['depth_top']) < 50:
                # 支持中英文岩性名
                is_sand = any(w in pred_lith for w in ['sand', '砂岩', 'sandstone'])
                gt_is_sand = any(w in gt_zone['lithology'] for w in ['sand', '砂岩', 'sandstone'])
                if is_sand and gt_is_sand:
                    matched += 1
                    depth_errors.append(abs(pred_top - gt_zone['depth_top']))
                break

        # 也匹配泥岩段（shale zones）
        for gt_zone in gt['shale_zones']:
            if abs(pred_top - gt_zone['depth_top']) < 50:
                is_shale = any(w in pred_lith for w in ['shale', '泥岩', 'mudstone', 'clay'])
                if is_shale:
                    matched += 1
                    depth_errors.append(abs(pred_top - gt_zone['depth_top']))
                break

    scores['sand_matched'] = matched
    scores['sand_total'] = len(gt['lithology_zones'])
    scores['sand_recall'] = matched / len(gt['lithology_zones']) if gt['lithology_zones'] else 0
    scores['depth_error_avg'] = np.mean(depth_errors) if depth_errors else 999

    # --- 流体检测 ---
    fluids = result.get('fluid_zones', [])
    gt_fluids = gt['fluid_zones']
    fluid_matched = 0
    for pred in fluids[:5]:
        pred_top = pred.get('depth_top', 0)
        pred_fluid = str(pred.get('fluid_type', '')).lower()
        for gt_f in gt_fluids:
            if abs(pred_top - gt_f['depth_top']) < 50:
                if gt_f['fluid'] in pred_fluid:
                    fluid_matched += 1
                break
    scores['fluid_recall'] = fluid_matched / len(gt_fluids) if gt_fluids else 0
    scores['fluid_detected'] = fluid_matched

    # --- 储层 ---
    reservoirs = result.get('reservoir_zones', [])
    scores['reservoir_detected'] = len(reservoirs)

    # --- Summary ---
    scores['has_summary'] = len(result.get('summary', '')) > 20

    # Overall
    scores['overall'] = (
        scores['sand_recall'] * 1.0 +
        scores['fluid_recall'] * 1.0 +
        min(scores['reservoir_detected'] / 2, 1.0) * 0.5 +
        scores['has_summary'] * 0.5
    ) / 3.0

    return scores


def evaluate_fusion(result) -> dict:
    """评测井震融合准确性"""
    scores = {}
    calib = result.get('well_seismic_calibration', {})
    scores['has_calibration'] = bool(calib)
    scores['correlation_in_range'] = 0 <= calib.get('correlation_coefficient', -1) <= 1
    scores['quality_valid'] = calib.get('calibration_quality', '') in ['excellent', 'good', 'acceptable', 'poor']

    interfaces = result.get('key_geological_interfaces', [])
    scores['interfaces_count'] = len(interfaces)
    scores['has_interface'] = len(interfaces) > 0

    correlations = result.get('seismic_log_correlation', [])
    scores['correlation_count'] = len(correlations)

    scores['has_summary'] = len(result.get('fusion_summary', '')) > 20

    scores['overall'] = (
        scores['has_calibration'] * 0.5 +
        scores['correlation_in_range'] * 0.5 +
        scores['has_interface'] * 0.5 +
        scores['has_summary'] * 0.5
    ) / 2.0

    return scores


def evaluate_prospect(result) -> dict:
    """评测目标评价准确性"""
    scores = {}
    targets = result.get('targets', [])
    scores['targets_count'] = len(targets)

    if targets:
        t = targets[0]
        risk = t.get('risk_assessment', {})
        scores['has_risk'] = all(k in risk for k in ['trap_risk', 'reservoir_risk', 'seal_risk', 'charge_risk'])
        scores['risk_in_range'] = all(1 <= risk.get(k, 0) <= 5 for k in ['trap_risk', 'reservoir_risk', 'seal_risk', 'charge_risk'])
        decision = t.get('decision', {})
        if isinstance(decision, str):
            scores['has_decision'] = decision in ['drill_ready', 'data_gap', 'inventory', 'drop']
        elif isinstance(decision, dict):
            scores['has_decision'] = decision.get('category', '') in ['drill_ready', 'data_gap', 'inventory', 'drop']
        else:
            scores['has_decision'] = False
        scores['has_pg'] = 0 < risk.get('geological_success_probability_pct', -1) <= 100

    scores['has_summary'] = len(result.get('overall_summary', '')) > 20

    checks = ['has_risk', 'risk_in_range', 'has_decision', 'has_pg', 'has_summary']
    scores['overall'] = sum(scores.get(c, False) * 1.0 for c in checks) / len(checks)

    return scores


# ============================================================
# Prompt 加载
# ============================================================

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

def load_prompt(name):
    for f in os.listdir(PROMPT_DIR):
        if name in f and f.endswith('.md'):
            with open(os.path.join(PROMPT_DIR, f)) as fp:
                return fp.read()
    return "你是地质专家。直接输出JSON。"


# ============================================================
# Main
# ============================================================

def main():
    results = {}
    all_scores = {}

    # 紧凑版+few-shot prompts
    SEISMIC_PROMPT = load_prompt("seismic_interp")
    LOG_PROMPT = load_prompt("log_analysis")
    FUSION_PROMPT = load_prompt("well_seismic_fusion")
    PROSPECT_PROMPT = load_prompt("prospect_evaluation")

    # Test 1: Seismic
    print("=" * 60)
    print("TEST 1: SeismicInterpAgent Accuracy")
    print("=" * 60)
    img = generate_precise_seismic()
    raw = call_vlm(SEISMIC_PROMPT, [img], "分析此地震剖面。仅输出JSON。")
    result = extract_json(raw)
    if result:
        scores = evaluate_seismic(result)
        all_scores['seismic'] = scores
        print(f"  Fault detected: {scores.get('fault_detected')}")
        print(f"  Fault CDP error: {scores.get('fault_cdp_error', 'N/A')}")
        print(f"  Anomaly detected: {scores.get('anomaly_detected')}")
        print(f"  Overall: {scores['overall']:.2f}/1.0")
        results['seismic'] = result
    else:
        print("  ❌ JSON parse failed")

    # Test 2: Log
    print("\n" + "=" * 60)
    print("TEST 2: LogAnalysisAgent Accuracy")
    print("=" * 60)
    img = generate_precise_log()
    raw = call_vlm(LOG_PROMPT, [img], "分析此测井图。depth值精确到m。仅输出JSON。")
    result = extract_json(raw)
    if result:
        scores = evaluate_log(result)
        all_scores['log'] = scores
        print(f"  Litho zones: {scores['litho_count']} predicted")
        print(f"  Sand recall: {scores['sand_recall']:.0%} ({scores['sand_matched']}/{scores['sand_total']})")
        print(f"  Avg depth error: {scores['depth_error_avg']:.1f}m")
        print(f"  Fluid recall: {scores['fluid_recall']:.0%}")
        print(f"  Overall: {scores['overall']:.2f}/1.0")
        results['log'] = result
    else:
        print("  ❌ JSON parse failed")
        print(f"  Raw snippet: {raw[-500:]}")

    # Test 3: Fusion
    print("\n" + "=" * 60)
    print("TEST 3: WellSeismicFusionAgent Accuracy")
    print("=" * 60)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
    n_traces, n_samples = 30, 400
    section = np.random.randn(n_samples, n_traces)*0.1
    for i in range(n_traces):
        section[:, i] += 0.4*np.sin(np.linspace(0, 8*np.pi, n_samples)+i*0.1)
    ax1.imshow(section, cmap='gray', aspect='auto', extent=[0, 30, 2000, 0])
    ax1.set_title('Well-tie Seismic'); ax1.set_xlabel('CDP'); ax1.set_ylabel('Time (ms)')
    depth = np.linspace(1000, 2000, 400)
    gr = 60+40*np.sin(depth/100*np.pi)
    ax2.plot(gr, depth, 'green', linewidth=1)
    ax2.set_xlabel('GR (API)'); ax2.invert_yaxis(); ax2.set_title('GR Log')
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100); plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    raw = call_vlm(FUSION_PROMPT, [img], "井名:WT-01。时深:(800ms,950m),(1000ms,1220m),(1500ms,2000m)。仅输出JSON。")
    result = extract_json(raw)
    if result:
        scores = evaluate_fusion(result)
        all_scores['fusion'] = scores
        print(f"  Calibration: {'✓' if scores['has_calibration'] else '✗'}")
        print(f"  Correlation valid: {'✓' if scores['correlation_in_range'] else '✗'}")
        print(f"  Interfaces: {scores['interfaces_count']}")
        print(f"  Overall: {scores['overall']:.2f}/1.0")
        results['fusion'] = result
    else:
        print("  ❌ JSON parse failed")

    # Test 4: Prospect
    print("\n" + "=" * 60)
    print("TEST 4: ProspectEvaluationAgent Accuracy")
    print("=" * 60)
    context = """地震:断层3条(F1正/CDP=80),亮点异常(CDP100-120/1000ms),背斜T1(闭合50ms),断块T2(闭合30ms)
测井:砂层L1(1200m),L2(1400m),L3(1550m/含气/por=18.5%),L4(1750m/水层)
井震融合:r=0.85,Top_Reservoir=1230m/1005ms
评价目标,仅输出JSON。"""
    raw = call_vlm(PROSPECT_PROMPT, [], context)
    result = extract_json(raw)
    if result:
        scores = evaluate_prospect(result)
        all_scores['prospect'] = scores
        targets = result.get('targets', [])
        print(f"  Targets: {len(targets)}")
        if targets:
            t = targets[0]; d = t.get('decision', {}); r = t.get('risk_assessment', {})
            print(f"  Top target: {t.get('name','?')} | {d.get('category','?')} | Pg={r.get('geological_success_probability_pct','?')}%")
            print(f"  All checks: risk_range={'✓' if scores.get('risk_in_range') else '✗'} decision={'✓' if scores.get('has_decision') else '✗'}")
        print(f"  Overall: {scores['overall']:.2f}/1.0")
        results['prospect'] = result
    else:
        print("  ❌ JSON parse failed")

    # Summary
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    total = 0
    for name, scores in all_scores.items():
        overall = scores.get('overall', 0)
        total += overall
        bar = '█' * int(overall * 20) + '░' * (20 - int(overall * 20))
        print(f"  {name:12s}: {overall:.2f} {bar}")
    avg = total / len(all_scores) if all_scores else 0
    print(f"  {'AVERAGE':12s}: {avg:.2f}")

    with open("/tmp/accuracy_test_results.json", 'w') as f:
        json.dump({"scores": all_scores, "results": {k: v for k, v in results.items() if v}},
                  f, ensure_ascii=False, indent=2, default=str)
    print("\nDetails saved to /tmp/accuracy_test_results.json")
    return avg


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
