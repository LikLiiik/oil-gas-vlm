"""
VLM Agent Prompt 测试脚本 (transformers 直接推理版本)

用法:
  cd /data/yxjiang/oil-gas-llm
  CUDA_VISIBLE_DEVICES=1 python test/test_live.py
"""

import json, base64, io, os, sys, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

# ============================================================
# Config
# ============================================================
MODEL_PATH = "/data/yxjiang/modelscope/hub/models/Qwen/Qwen3.5-9B"
PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print("Loading model...")

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Model loaded: {type(model).__name__}")

# ============================================================
# 模拟数据生成
# ============================================================

def generate_seismic_section(n_traces=200, n_samples=400) -> Image.Image:
    """生成模拟地震剖面图（含断层+亮点）, 返回 PIL Image"""
    t = np.linspace(0, 2.0, n_samples) * 2 * np.pi
    section = np.zeros((n_samples, n_traces))
    for i in range(n_traces):
        trace = (
            0.5 * np.sin(t * 0.3 + i * 0.02) +
            0.3 * np.sin(t * 0.7 + i * 0.05) +
            0.2 * np.sin(t * 1.2 + i * 0.03) +
            0.15 * np.random.randn(n_samples)
        )
        # 断层在第80道
        if 75 < i < 85:
            shift = int(5 * (1 - abs(i - 80) / 5))
            trace = np.roll(trace, shift)
        section[:, i] = trace
    # 亮点异常（1200ms处）
    section[200:210, 100:120] *= 3.0

    fig, ax = plt.subplots(figsize=(10, 6))
    vmin, vmax = np.percentile(section, [5, 95])
    ax.imshow(section, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax,
              extent=[1, n_traces, 2000, 0])
    ax.set_xlabel('CDP'); ax.set_ylabel('Time (ms)')
    ax.set_title('Seismic Inline Section (Simulated)')
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def generate_log_curves() -> Image.Image:
    """生成模拟测井曲线图, 返回 PIL Image"""
    depth = np.linspace(1000, 2000, 2000)
    gr = np.full(2000, 90.0)
    rt = np.full(2000, 3.0)
    ac = np.full(2000, 75.0)
    den = np.full(2000, 2.55)
    cnl = np.full(2000, 0.30)
    sp = np.full(2000, 0.0)

    sand_intervals = [(1200, 1250), (1400, 1430), (1550, 1620), (1750, 1800)]
    for top, bot in sand_intervals:
        m = (depth >= top) & (depth <= bot)
        n = m.sum()
        gr[m] = 35 + np.random.randn(n)*5
        ac[m] = 62 + np.random.randn(n)*3
        den[m] = 2.30 + np.random.randn(n)*0.05
        cnl[m] = 0.15 + np.random.randn(n)*0.03
        sp[m] = -30 + np.random.randn(n)*5

    gas_zone = (1550, 1600)
    m = (depth >= gas_zone[0]) & (depth <= gas_zone[1])
    rt[m] = 80 + np.random.randn(m.sum())*10

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

def load_prompt(agent_name: str) -> str:
    for f in os.listdir(PROMPT_DIR):
        if agent_name in f and f.endswith('.md'):
            with open(os.path.join(PROMPT_DIR, f)) as fp:
                return fp.read()
    return ""


def call_vlm(system_prompt: str, images: list, user_text: str,
             max_tokens=32768, temperature=0.3) -> str:
    """调用VLM, 使用 transformers。max_tokens=32768确保JSON完整"""
    # 构建消息
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
        # 纯文本输入
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

    # 跳过 <think> 块，只取模型实际输出
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()

    return response


def extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return None


# ============================================================
# 测试用例
# ============================================================

def test_seismic():
    print("\n" + "=" * 60)
    print("TEST 1: SeismicInterpAgent")
    print("=" * 60)
    sys_prompt = load_prompt("seismic_interp")
    img = generate_seismic_section()
    user_text = "分析此地震剖面，识别断层、层位和异常体。仅输出JSON。"

    print("Calling VLM...")
    raw = call_vlm(sys_prompt, [img], user_text)
    print(f"\n--- RAW OUTPUT ---\n{raw[:1000]}\n")

    result = extract_json(raw)
    if result:
        print("--- PARSED ---")
        for k in ['faults', 'horizons', 'seismic_facies', 'anomalies', 'summary']:
            v = result.get(k)
            if isinstance(v, list): print(f"  {k}: {len(v)} items")
            elif isinstance(v, str): print(f"  {k}: {v[:120]}...")
    else:
        print("[WARN] JSON parse failed, raw text returned")
    return result


def test_log():
    print("\n" + "=" * 60)
    print("TEST 2: LogAnalysisAgent")
    print("=" * 60)
    sys_prompt = load_prompt("log_analysis")
    img = generate_log_curves()
    user_text = "分析此测井曲线图，输出JSON含lithology_zones/reservoir_zones/fluid_zones/summary。"

    print("Calling VLM...")
    raw = call_vlm(sys_prompt, [img], user_text)
    print(f"\n--- RAW OUTPUT ---\n{raw[:1000]}\n")

    result = extract_json(raw)
    if result:
        print("--- PARSED ---")
        for k in ['lithology_zones', 'reservoir_zones', 'fluid_zones', 'summary']:
            v = result.get(k)
            if isinstance(v, list):
                print(f"  {k}: {len(v)} items")
                for item in v[:2]:
                    print(f"    - {json.dumps(item, ensure_ascii=False)[:150]}")
            elif isinstance(v, str): print(f"  {k}: {v[:120]}...")
    else:
        print("[WARN] JSON parse failed")
    return result


def test_fusion():
    print("\n" + "=" * 60)
    print("TEST 3: WellSeismicFusionAgent")
    print("=" * 60)
    sys_prompt = load_prompt("well_seismic_fusion")

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

    user_text = "井名:Well-Test-01。时深:(800ms,950m),(1000ms,1220m),(1500ms,2000m)。分析井震关系，仅输出JSON。"

    print("Calling VLM...")
    raw = call_vlm(sys_prompt, [img], user_text)
    print(f"\n--- RAW OUTPUT ---\n{raw[:1000]}\n")

    result = extract_json(raw)
    if result:
        print("--- PARSED ---")
        for k in ['well_seismic_calibration', 'key_geological_interfaces', 'fusion_summary']:
            v = result.get(k)
            if isinstance(v, dict):
                print(f"  {k}: {json.dumps(v, ensure_ascii=False)[:200]}")
            elif isinstance(v, str): print(f"  {k}: {v[:150]}...")
            elif isinstance(v, list): print(f"  {k}: {len(v)} items")
    else:
        print("[WARN] JSON parse failed")
    return result


def test_prospect():
    print("\n" + "=" * 60)
    print("TEST 4: ProspectEvaluationAgent")
    print("=" * 60)
    sys_prompt = load_prompt("prospect_evaluation")

    context = """## 地震解释结果
识别断层3条(F1正断层/NE倾, F2逆断层/NW倾, F3走滑断层)，追踪层位4个(H1不整合面, H2-H4层序界面)。
发现亮点异常1个(A1型/1200ms)，构造圈闭2个(T1背斜/闭合50ms, T2断块/闭合30ms)。

## 测井分析结果
Well-Test-01分析: 钻遇4套砂层(L1 1200-1250m细砂岩, L2 1400-1430m粉砂岩, L3 1550-1620m中砂岩, L4 1750-1800m含砾砂岩)。
R1储层1550-1600m/por=18.5%/含气/RT高值85Ω·m。R2储层1750-1780m/水层。

## 井震融合结果
井震相关系数0.85/质量good。关键界面: I1(Top_Reservoir/1230m/1005ms/强波峰)。

评价以上勘探目标，仅输出JSON。"""

    print("Calling VLM...")
    raw = call_vlm(sys_prompt, [], context)
    print(f"\n--- RAW OUTPUT ---\n{raw[:1200]}\n")

    result = extract_json(raw)
    if result:
        print("--- PARSED ---")
        targets = result.get('targets', [])
        print(f"  targets: {len(targets)} prospects")
        for t in targets[:4]:
            d = t.get('decision', {})
            print(f"    rank={t.get('priority_rank','?')} | {t.get('name','?')} | {d.get('category','?')}")
        risk = result.get('risk_summary', {})
        print(f"  risk_summary: {risk}")
        s = result.get('overall_summary', '')
        print(f"  summary: {s[:200]}...")
    else:
        print("[WARN] JSON parse failed")
    return result


if __name__ == "__main__":
    results = {}
    results['seismic'] = test_seismic()
    results['log'] = test_log()
    results['fusion'] = test_fusion()
    results['prospect'] = test_prospect()

    print("\n" + "=" * 60)
    print("SUMMARY")
    for name, r in results.items():
        print(f"  {name}: {'PASS' if r else 'FAIL (no JSON)'}")

    # Save
    with open("/tmp/vlm_test_results.json", 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print("Results saved to /tmp/vlm_test_results.json")
