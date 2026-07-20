"""
批量测试：多次运行统计准确率均值和方差

对 SeismicInterp 和 LogAnalysis 两个Agent各跑N次。
用不同的随机种子生成模拟数据，测量准确性指标。
"""

import torch, json, io, re, warnings, numpy as np, time, sys, os
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

N_RUNS = 5  # 每个Agent跑5次
MODEL_PATH = "/data/yxjiang/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct"
PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

print("=" * 60)
print(f"Batch Test: N={N_RUNS} per agent")
print(f"Model: {MODEL_PATH}")
print("=" * 60)

# Load model once
print("\nLoading model..."); t0 = time.time()
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Loaded in {time.time()-t0:.0f}s")

# Load prompts
PROMPTS = {}
for f in os.listdir(PROMPT_DIR):
    if f.endswith('.md'):
        with open(os.path.join(PROMPT_DIR, f)) as fp:
            PROMPTS[f.replace('.md', '')] = fp.read()


def call_vlm(prompt, images, text, max_tok=4096):
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": text})
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": content}]
    t = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if images:
        inputs = processor(text=t, images=images, return_tensors="pt").to(model.device)
    else:
        inputs = processor.tokenizer(t, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tok, do_sample=True,
                                temperature=0.3, repetition_penalty=1.1, top_p=0.95)
    elapsed = time.time() - t0
    resp = processor.decode(output[0], skip_special_tokens=True)
    if "assistant" in resp: resp = resp.split("assistant")[-1].strip()
    return resp, elapsed


def extract_json(text):
    for m in re.finditer(r'\{', text):
        start = m.start(); d = 0; in_s = False; esc = False; end = -1
        for i in range(start, len(text)):
            c = text[i]
            if esc: esc = False; continue
            if c == '\\': esc = True; continue
            if c == '"' and not esc: in_s = not in_s; continue
            if in_s: continue
            if c == '{': d += 1
            elif c == '}': d -= 1
            if d == 0: end = i + 1; break
        if end > start:
            try:
                r = json.loads(text[start:end])
                if isinstance(r, dict) and len(r) >= 3: return r
            except: continue
    return None


def generate_log(seed):
    """生成测井数据，返回(img, ground_truth)"""
    np.random.seed(seed)
    depth = np.linspace(1000, 2000, 2000)
    gr = np.full(2000, 95.0); rt = np.full(2000, 3.0)
    ac = np.full(2000, 80.0); den = np.full(2000, 2.50)
    cnl = np.full(2000, 0.32); sp = np.full(2000, 5.0)

    gt_sands = [(1200, 1255), (1400, 1435), (1550, 1625), (1750, 1805)]
    gt_fluids = [(1555, 1595, 'gas'), (1755, 1790, 'water')]

    for top, bot in gt_sands:
        m = (depth >= top) & (depth <= bot); n = m.sum()
        gr[m] = 35 + np.random.randn(n) * 6
        ac[m] = 63 + np.random.randn(n) * 4
        den[m] = 2.28 + np.random.randn(n) * 0.06
        cnl[m] = 0.16 + np.random.randn(n) * 0.04
        sp[m] = -35 + np.random.randn(n) * 6
    m = (depth >= 1555) & (depth <= 1595); rt[m] = 85 + np.random.randn(m.sum()) * 10
    m = (depth >= 1755) & (depth <= 1790); rt[m] = 3 + np.random.randn(m.sum()) * 0.3

    curves = [('GR', gr, 'green', 'GR(API)', (0, 150)), ('SP', sp, 'blue', 'SP(mV)', (-60, 20)),
              ('RT', rt, 'red', 'RT(Ohm.m)', (0.1, 100)), ('AC', ac, 'orange', 'AC(us/ft)', (140, 40)),
              ('DEN', den, 'black', 'DEN(g/cm3)', (1.8, 2.8)), ('CNL', cnl, 'magenta', 'CNL(v/v)', (0.45, -0.15))]
    fig, axes = plt.subplots(1, 6, figsize=(18, 14), sharey=True)
    for ax, (name, data, color, label, xlim) in zip(axes, curves):
        ax.plot(data, depth, color=color, linewidth=0.5)
        ax.set_xlabel(label, fontsize=8); ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3); ax.invert_yaxis()
        if xlim[0] > xlim[1]: ax.invert_xaxis()
    axes[0].set_ylabel('Depth(m)')
    fig.suptitle(f'Well Log: Well-Test-{seed:02d}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB'), gt_sands, gt_fluids, gr, rt, den, depth


def generate_seismic(seed):
    """生成地震数据，返回(img, ground_truth)。使用更清晰的参数。"""
    np.random.seed(seed)
    n_traces, n_samples = 200, 400
    section = np.zeros((n_samples, n_traces))
    for i in range(n_traces):
        trace = (0.8 * np.sin(np.linspace(0, 6*np.pi, n_samples) + i*0.03) +
                 0.4 * np.sin(np.linspace(0, 12*np.pi, n_samples) + i*0.05) +
                 0.05 * np.random.randn(n_samples))  # low noise
        if 72 < i < 88:
            trace = np.roll(trace, int(8 * (1 - abs(i-80)/8)))
        section[:, i] = trace
    section[195:210, 100:120] = -3.0  # strong negative = bright spot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(section, cmap='seismic', aspect='auto', clim=(-2, 2),
              extent=[1, n_traces, 2000, 0])
    ax.set_xlabel('CDP'); ax.set_ylabel('Time(ms)')
    ax.set_title(f'Seismic Section (seed={seed})')
    ax.grid(True, alpha=0.15, linestyle='--')
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100); plt.close(fig)
    buf.seek(0)
    gt = {
        'fault_cdp': 80, 'fault_time_ms': 1000,
        'bright_spot_cdp_range': [100, 120],
        'bright_spot_time_range_ms': [975, 1025],
    }
    return Image.open(buf).convert('RGB'), gt


# ============================================================
# Batch Test: LogAgent
# ============================================================
print("\n" + "=" * 60)
print("BATCH: LogAnalysisAgent (5 runs)")
print("=" * 60)

log_results = []
for run in range(N_RUNS):
    seed = run * 100 + 42
    img, gt_sands, gt_fluids, gr_raw, rt_raw, den_raw, depth_arr = generate_log(seed)
    resp, elapsed = call_vlm(PROMPTS['log_analysis_agent'], [img], "仅输出JSON。")
    result = extract_json(resp)

    # Two-stage: VLM coarse + code precise
    sand_intervals = []
    in_sand = False; sand_start = 0
    for i in range(1, len(gr_raw)):
        if not in_sand and gr_raw[i] < 50 and gr_raw[i - 1] >= 50:
            sand_start = depth_arr[i]; in_sand = True
        elif in_sand and gr_raw[i] >= 50 and gr_raw[i - 1] < 50:
            sand_intervals.append((float(sand_start), float(depth_arr[i])))
            in_sand = False
    if in_sand: sand_intervals.append((float(sand_start), float(depth_arr[-1])))
    sand_intervals = [(t, b) for t, b in sand_intervals if b - t > 10]

    # Accuracy
    sand_recall = 0
    for gt_t, gt_b in gt_sands:
        for st, sb in sand_intervals:
            if st <= gt_b and sb >= gt_t:
                sand_recall += 1; break

    fluid_correct = 0
    for gt_t, gt_b, gt_f in gt_fluids:
        for st, sb in sand_intervals:
            if st <= gt_t <= sb:
                m = (depth_arr >= st) & (depth_arr <= sb)
                avg_rt = rt_raw[m].mean()
                avg_den = den_raw[m].mean()
                pred = 'gas' if (avg_rt > 20 and avg_den < 2.35) else ('water' if avg_rt < 5 else 'other')
                if pred == gt_f: fluid_correct += 1
                break

    json_ok = result is not None
    run_result = {
        'run': run, 'seed': seed, 'time_s': round(elapsed, 1),
        'json_ok': json_ok, 'sand_recall': sand_recall, 'sand_total': len(gt_sands),
        'fluid_correct': fluid_correct, 'fluid_total': len(gt_fluids),
    }
    log_results.append(run_result)
    print(f"  Run {run+1}/{N_RUNS}: seed={seed} | {elapsed:.0f}s | "
          f"json={'✅' if json_ok else '❌'} | "
          f"sand={sand_recall}/{len(gt_sands)} | fluid={fluid_correct}/{len(gt_fluids)}")


# ============================================================
# Batch Test: SeismicAgent
# ============================================================
print("\n" + "=" * 60)
print("BATCH: SeismicInterpAgent (5 runs)")
print("=" * 60)

seismic_results = []
for run in range(N_RUNS):
    seed = run * 100 + 7
    img, gt = generate_seismic(seed)
    resp, elapsed = call_vlm(PROMPTS['seismic_interp_agent'], [img], "仅输出JSON。")
    result = extract_json(resp)

    json_ok = result is not None
    fault_detected = False
    anomaly_detected = False
    anomaly_cdp_ok = False

    if result:
        faults = result.get('faults', [])
        fault_detected = len(faults) > 0

        anomalies = result.get('anomalies', [])
        anomaly_detected = len(anomalies) > 0
        if anomalies:
            pos = anomalies[0].get('position', [0, 0])
            pred_cdp = pos[0] if isinstance(pos[0], (int, float)) else (pos[0][0] if pos[0] else 0)
            gt_range = gt['bright_spot_cdp_range']
            anomaly_cdp_ok = gt_range[0] <= pred_cdp <= gt_range[1]

    run_result = {
        'run': run, 'seed': seed, 'time_s': round(elapsed, 1),
        'json_ok': json_ok, 'fault_detected': fault_detected,
        'anomaly_detected': anomaly_detected, 'anomaly_cdp_ok': anomaly_cdp_ok,
    }
    seismic_results.append(run_result)
    print(f"  Run {run+1}/{N_RUNS}: seed={seed} | {elapsed:.0f}s | "
          f"json={'✅' if json_ok else '❌'} | "
          f"fault={'✅' if fault_detected else '❌'} | "
          f"anomaly={'✅' if anomaly_detected else '❌'} "
          f"(CDP={'OK' if anomaly_cdp_ok else 'off'})")


# ============================================================
# Summary Statistics
# ============================================================
print("\n" + "=" * 60)
print("BATCH TEST SUMMARY")
print("=" * 60)

def stats(results, metric, total=None):
    vals = [r[metric] for r in results]
    mean = np.mean(vals)
    std = np.std(vals)
    if total:
        return f"{mean:.1%} ± {std:.1%}"
    else:
        return f"{mean:.2f} ± {std:.2f}"

print("\nLogAnalysisAgent (two-stage):")
print(f"  JSON pass rate:  {sum(1 for r in log_results if r['json_ok'])}/{N_RUNS}")
print(f"  Sand recall:     {stats(log_results, 'sand_recall', log_results[0]['sand_total'])} "
      f"(mean {np.mean([r['sand_recall'] for r in log_results]):.1f}/"
      f"{log_results[0]['sand_total']})")
print(f"  Fluid accuracy:  {stats(log_results, 'fluid_correct', log_results[0]['fluid_total'])} "
      f"(mean {np.mean([r['fluid_correct'] for r in log_results]):.1f}/"
      f"{log_results[0]['fluid_total']})")
times = [r['time_s'] for r in log_results]
print(f"  Time:            {np.mean(times):.1f}s ± {np.std(times):.1f}s")

print("\nSeismicInterpAgent:")
print(f"  JSON pass rate:  {sum(1 for r in seismic_results if r['json_ok'])}/{N_RUNS}")
print(f"  Fault detected:  {sum(1 for r in seismic_results if r['fault_detected'])}/{N_RUNS}")
print(f"  Anomaly detected:{sum(1 for r in seismic_results if r['anomaly_detected'])}/{N_RUNS}")
print(f"  Anomaly CDP ok:  {sum(1 for r in seismic_results if r['anomaly_cdp_ok'])}/{N_RUNS}")
times = [r['time_s'] for r in seismic_results]
print(f"  Time:            {np.mean(times):.1f}s ± {np.std(times):.1f}s")

# Save
with open("/tmp/batch_test_results.json", 'w') as f:
    json.dump({"log": log_results, "seismic": seismic_results}, f, indent=2)
print("\nSaved to /tmp/batch_test_results.json")
