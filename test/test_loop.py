"""
闭环Agent工作流 — VLM Planning → 下游执行 → VLM验证 → 迭代收敛

完整实现 SeismicInterpAgent 和 LogAnalysisAgent 的闭环。
下游模型用模拟替代（真实使用时替换为 YOLO-World / SAM / 代码的API调用）。
"""

import torch, json, io, re, warnings, numpy as np, time, sys, os
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MAX_ITERATIONS = 3  # 最多迭代3轮
MODEL_PATH = "/data/yxjiang/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct"

print("Loading VLM...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)


# ============================================================
# VLM 调用
# ============================================================

def call_vlm(system_prompt: str, images: list, user_text: str,
             max_tokens=4096, temperature=0.3) -> str:
    content = [{"type": "image", "image": img} for img in images]
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
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                                temperature=temperature, repetition_penalty=1.1, top_p=0.95)
    elapsed = time.time() - t0
    resp = processor.decode(output[0], skip_special_tokens=True)
    if "assistant" in resp: resp = resp.split("assistant")[-1].strip()
    return resp, elapsed


def extract_json(text: str) -> dict | None:
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
                if isinstance(r, dict) and len(r) >= 2: return r
            except: continue
    return None


# ============================================================
# 模拟下游模型 (真实使用时替换为 YOLO-World / SAM / 代码)
# ============================================================

class MockDownstreamModels:
    """模拟下游模型。真实使用时替换为实际API调用。"""

    @staticmethod
    def yolo_world_detect(image: Image.Image, categories: list) -> list:
        """
        模拟 YOLO-World 检测。
        真实: model.detect(image, categories) → [{bbox, class_name, confidence}]
        这里: 基于类别名和expected_range生成模拟bbox
        """
        results = []
        for cat in categories:
            cdp_range = cat.get('expected_cdp_range', [0, 300])
            time_range = cat.get('expected_time_range_ms', [0, 2500])
            threshold = cat.get('confidence_threshold', 0.3)
            class_name = cat.get('class_name', 'unknown')

            if 'fault' in class_name.lower():
                # 模拟: 在搜索范围内检测到1-2个断层候选
                cdp_center = (cdp_range[0] + cdp_range[1]) / 2
                results.append({
                    "id": f"det_fault_1",
                    "class_name": class_name,
                    "bbox": [cdp_center - 6, cdp_center + 7,
                             time_range[0] + 100, time_range[0] + 400],
                    "confidence": 0.45,
                    "source": "yolo_world_simulated",
                })
                # 加一个假阳性（在范围边缘）
                if cdp_range[1] - cdp_range[0] > 50:
                    results.append({
                        "id": f"det_fault_2",
                        "class_name": class_name,
                        "bbox": [cdp_range[1] - 10, cdp_range[1],
                                 time_range[0] + 500, time_range[0] + 560],
                        "confidence": 0.32,
                        "source": "yolo_world_simulated",
                    })

            elif 'bright' in class_name.lower() or 'spot' in class_name.lower():
                results.append({
                    "id": "det_bright_1",
                    "class_name": class_name,
                    "bbox": [cdp_range[0] - 10, cdp_range[1] + 5,
                             time_range[0] - 30, time_range[1] + 30],
                    "confidence": 0.88,
                    "source": "yolo_world_simulated",
                })

            elif 'channel' in class_name.lower():
                results.append({
                    "id": "det_channel_1",
                    "class_name": class_name,
                    "bbox": [cdp_range[0], cdp_range[1],
                             time_range[0] - 20, time_range[1] + 20],
                    "confidence": 0.72,
                    "source": "yolo_world_simulated",
                })

        # 只有高于阈值的才返回
        return [r for r in results if r['confidence'] >= threshold]

    @staticmethod
    def traditional_code_execute(curve_rules: list, raw_data: dict) -> list:
        """
        模拟传统代码执行阈值检测。
        真实: 用GR/RT numpy数据 + 阈值规则 → 精确边界（±0.1m）
        这里: 模拟返回精确区间
        """
        results = []
        for rule in curve_rules:
            for depth_range in rule.get('expected_depth_ranges', []):
                top = depth_range['top_m']
                bot = depth_range['bottom_m']
                rule_str = rule.get('rule', '')
                class_name = rule.get('class_name', 'unknown')

                # 模拟精确边界: 在VLM给的大致范围内加一点随机扰动
                exact_top = top + np.random.randn() * 3
                exact_bot = bot + np.random.randn() * 3
                # 模拟噪音导致的分段
                segments = [(round(exact_top, 1), round(exact_bot, 1))]
                if bot - top > 30:  # 厚层可能会被薄夹层打断
                    mid = (top + bot) / 2
                    segments = [
                        (round(exact_top, 1), round(mid - 1 + np.random.randn(), 1)),
                        (round(mid + 1 + np.random.randn(), 1), round(exact_bot, 1)),
                    ]

                for seg_top, seg_bot in segments:
                    results.append({
                        "id": f"code_{class_name}_{seg_top}",
                        "class_name": class_name,
                        "rule": rule_str,
                        "depth_top_m": seg_top,
                        "depth_bottom_m": seg_bot,
                        "source": "traditional_code_simulated",
                    })
        return results


# ============================================================
# 真实数据生成
# ============================================================

def generate_seismic(seed=42):
    """生成逼真模拟地震剖面"""
    np.random.seed(seed)
    n_traces, n_samples = 300, 500
    reflectivity = np.zeros((n_samples, n_traces))
    for layer_idx in range(8):
        base_depth = 50 + layer_idx * 55
        fold_amp = 15 + np.random.rand() * 20
        for i in range(n_traces):
            depth = base_depth + fold_amp * np.sin(i / (80 + np.random.rand()*40) * 2 * np.pi)
            d_idx = int(np.clip(depth + np.random.randn()*3, 0, n_samples-1))
            if d_idx < n_samples:
                reflectivity[d_idx, i] = np.random.rand()*0.6 + 0.2

    # 断层 CDP=120, throw=15
    reflectivity[:, 120:] = np.roll(reflectivity[:, 120:], 15, axis=0)
    # 亮点 CDP 180-210, time 1200-1300ms
    reflectivity[240:243, 180:210] = -1.5
    # 河道 CDP 50-80, time 1500-1540
    for i in range(50, 80):
        thickness = int(8*(1-((i-65)/15)**2))
        if thickness > 0:
            reflectivity[300:300+thickness, i] = 0.4 + np.random.randn(thickness)*0.15

    def ricker(length, dt, f):
        t = np.arange(length)*dt - (length*dt)/2
        return (1-2*(np.pi*f*t)**2)*np.exp(-(np.pi*f*t)**2)
    wavelet = ricker(64, 0.004, 25)
    seismic = np.zeros_like(reflectivity)
    for i in range(n_traces):
        seismic[:, i] = np.convolve(reflectivity[:, i], wavelet, mode='same')
    seismic += np.random.randn(*seismic.shape)*0.03

    fig, ax = plt.subplots(figsize=(14, 10))
    vmax = np.percentile(np.abs(seismic), 98)
    ax.imshow(seismic, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax,
              extent=[1, n_traces, 2500, 0])
    ax.set_xlabel('CDP'); ax.set_ylabel('Two-Way Time (ms)')
    ax.set_title('Seismic Inline Section')
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB'), seismic


def generate_log(seed=42):
    """生成测井数据"""
    np.random.seed(seed)
    depth = np.linspace(1000, 2000, 2000)
    gr = np.full(2000, 95.0); rt = np.full(2000, 3.0)
    ac = np.full(2000, 80.0); den = np.full(2000, 2.50)
    cnl = np.full(2000, 0.32); sp = np.full(2000, 5.0)
    for top, bot in [(1200,1255),(1400,1435),(1550,1625),(1750,1805)]:
        m = (depth>=top)&(depth<=bot); n=m.sum()
        gr[m]=35+np.random.randn(n)*6; ac[m]=63+np.random.randn(n)*4
        den[m]=2.28+np.random.randn(n)*0.06; cnl[m]=0.16+np.random.randn(n)*0.04
        sp[m]=-35+np.random.randn(n)*6
    m=(depth>=1555)&(depth<=1595); rt[m]=85+np.random.randn(m.sum())*10
    m=(depth>=1755)&(depth<=1790); rt[m]=3+np.random.randn(m.sum())*0.3

    curves = [('GR',gr,'green','GR(API)',(0,150)),('SP',sp,'blue','SP(mV)',(-60,20)),
              ('RT',rt,'red','RT(Ohm.m)',(0.1,100)),('AC',ac,'orange','AC(us/ft)',(140,40)),
              ('DEN',den,'black','DEN(g/cm3)',(1.8,2.8)),('CNL',cnl,'magenta','CNL(v/v)',(0.45,-0.15))]
    fig,axes=plt.subplots(1,6,figsize=(18,14),sharey=True)
    for ax,(name,data,color,label,xlim) in zip(axes,curves):
        ax.plot(data,depth,color=color,linewidth=0.5)
        ax.set_xlabel(label,fontsize=8); ax.set_xlim(xlim)
        ax.grid(True,alpha=0.3); ax.invert_yaxis()
        if xlim[0]>xlim[1]: ax.invert_xaxis()
    axes[0].set_ylabel('Depth(m)')
    fig.suptitle('Well Log: Well-Test-01',fontsize=12,fontweight='bold')
    plt.tight_layout()
    buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=120); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB'), {'depth':depth,'gr':gr,'rt':rt,'den':den,'ac':ac,'cnl':cnl,'sp':sp}


# ============================================================
# 闭环工作流: SeismicInterpAgent
# ============================================================

def run_seismic_loop(seismic_img):
    """SeismicInterpAgent 完整闭环: Plan → Detect → Verify → Refine"""

    planning_prompt = """你是地球物理专家。分析地震剖面（红/暖=波峰，蓝/冷=波谷）。
输出所有需要下游模型检测的目标。每类包含: class_name, description, expected_cdp_range, expected_time_range_ms, confidence_threshold, max_detections。
目标类型: fault plane(同相轴错断), bright spot anomaly(强负振幅亮点), channel(透镜状反射)
JSON格式:
{"downstream_prompts":{"yolo_world":{"categories":[...]}},"analysis":{"summary":"..."}}"""

    verification_prompt = """你是地球物理专家。下游YOLO-World已完成检测，请验证每条结果。

原始下游检测指令: {plan}
下游检测到的候选: {detections}

请重新查看原始地震剖面图像，逐一判断:
1. 每条检测是真实地质特征还是假阳性？
2. 如果有假阳性，原因是什么？（河道边缘？噪声？）
3. 有没有遗漏的目标？剖面中可以看到但下游没检测到的？
4. 如果有遗漏或假阳性，给出修正后的下游检测指令

JSON:
{"verified":[{"id":"...","is_real":true/false,"confidence":0.9,"geological_reason":"...","rejection_reason":null}],
 "false_positives_removed":0,"missed_targets":[],"converged":true/false,
 "refined_prompts":{...},"final_summary":"..."}"""

    # ====== Round 1: Planning ======
    print("\n--- Round 1: VLM Planning ---")
    resp, t = call_vlm(planning_prompt, [seismic_img], "分析地震剖面，输出下游检测指令。仅输出JSON。")
    plan = extract_json(resp)
    if not plan:
        print(f"  ❌ Planning failed. Raw: {resp[:300]}")
        return None
    print(f"  ✅ Plan generated ({t:.0f}s)")

    categories = plan.get('downstream_prompts', {}).get('yolo_world', {}).get('categories', [])
    analysis = plan.get('analysis', {})
    print(f"  Categories: {len(categories)}")
    for c in categories:
        print(f"    {c.get('class_name')}: CDP{c.get('expected_cdp_range')} time{c.get('expected_time_range_ms')}ms")

    # ====== Round 1: Downstream Execution ======
    print("\n--- Round 1: Downstream Execution (simulated YOLO-World) ---")
    detections = MockDownstreamModels.yolo_world_detect(seismic_img, categories)
    print(f"  YOLO returned {len(detections)} detections")
    for d in detections:
        print(f"    {d['id']}: {d['class_name']} bbox={d['bbox']} conf={d['confidence']}")

    # ====== Round 1: Verification ======
    print("\n--- Round 1: VLM Verification ---")
    user_text = f"Plan: {json.dumps(categories, ensure_ascii=False)}\nDetections: {json.dumps(detections, ensure_ascii=False)}\n验证每条检测结果。仅输出JSON。"
    resp, t = call_vlm(verification_prompt, [seismic_img], user_text)
    verification = extract_json(resp)
    if not verification:
        print(f"  ❌ Verification failed. Raw: {resp[:300]}")
        return plan
    print(f"  ✅ Verification complete ({t:.0f}s)")

    verified = verification.get('verified', [])
    real_count = sum(1 for v in verified if v.get('is_real'))
    false_count = sum(1 for v in verified if not v.get('is_real'))
    missed = verification.get('missed_targets', [])
    print(f"  Real: {real_count} | False positive: {false_count} | Missed: {len(missed)}")
    for v in verified:
        status = "✅" if v.get('is_real') else "❌"
        print(f"    {status} {v.get('id')}: {v.get('geological_reason','')[:80]}")

    # ====== Round 2: Refine (if needed) ======
    if not verification.get('converged', True) and missed:
        print("\n--- Round 2: Refined Detection ---")
        refined = verification.get('refined_prompts', {})
        refined_cats = refined.get('yolo_world', {}).get('categories', [])
        if refined_cats:
            print(f"  Refined categories: {len(refined_cats)}")
            detections2 = MockDownstreamModels.yolo_world_detect(seismic_img, refined_cats)
            print(f"  YOLO returned {len(detections2)} additional detections")

            # Second verification
            user_text2 = f"Refined detections: {json.dumps(detections2, ensure_ascii=False)}\n验证追加的检测结果。仅输出JSON。"
            resp2, t2 = call_vlm(verification_prompt, [seismic_img], user_text2)
            verification2 = extract_json(resp2)
            if verification2:
                verified2 = verification2.get('verified', [])
                verified.extend(verified2)
                print(f"  ✅ Round 2 complete ({t2:.0f}s): {len(verified2)} more verified")

    # ====== Final Output ======
    final = {
        "planning": plan,
        "verification": verification,
        "final_verified": [v for v in verified if v.get('is_real')],
        "false_positives_removed": [v for v in verified if not v.get('is_real')],
        "analysis": analysis,
    }
    print(f"\n  Final: {len(final['final_verified'])} verified targets, "
          f"{len(final['false_positives_removed'])} false positives removed")
    return final


# ============================================================
# 闭环工作流: LogAnalysisAgent
# ============================================================

def run_log_loop(log_img, raw_data):
    """LogAnalysisAgent 闭环: Plan → Code Execute → Verify → Adjust"""

    planning_prompt = """你是测井专家。分析测井曲线图。输出传统代码的阈值检测指令。
识别: GR<50=砂岩, GR>75=泥岩, RT>20+低DEN+低CNL=含气, RT<5=水层。
每类输出: class_name, rule, expected_depth_ranges。
JSON:
{"downstream_prompts":{"traditional_code":{"curves":[{"class_name":"low_GR_sandstone","rule":"GR<50",
 "expected_depth_ranges":[{"top_m":1200,"bottom_m":1260}]}]}},"analysis":{"lithology_summary":"..."}}"""

    verification_prompt = """你是测井专家。传统代码已根据你的指令完成了阈值检测。请验证结果。

原始指令: {plan}
代码检测结果: {code_results}

请查看原始测井曲线图，验证每条代码输出:
1. 相邻段间距<2m的是噪音还是真实夹层？应该合并吗？
2. 边界处GR是否确实穿越了阈值？RT/DEN/CNL有同步变化吗？
3. 流体判断(含气/水层)是否正确？
4. 有没有遗漏的薄层？

JSON:
{"verified":[{"id":"...","is_real":true,"adjusted_top":1200.1,"adjusted_bottom":1255.1,
   "adjustment":"合并0.5m噪音间隙"}],
 "interval_merges":[{"merge":["seg1_id","seg2_id"],"reason":"0.5m间隙为噪声"}],
 "fluid_verified":[{"id":"...","fluid_ok":true,"actual_fluid":"gas"}],
 "converged":true,"final_summary":"..."}"""

    # ====== Round 1: Planning ======
    print("\n--- Round 1: VLM Planning ---")
    resp, t = call_vlm(planning_prompt, [log_img], "分析测井曲线，输出阈值指令。仅输出JSON。")
    plan = extract_json(resp)
    if not plan:
        print(f"  ❌ Planning failed")
        return None
    print(f"  ✅ Plan generated ({t:.0f}s)")

    curves = plan.get('downstream_prompts', {}).get('traditional_code', {}).get('curves', [])
    print(f"  Rules: {len(curves)}")
    for c in curves:
        for r in c.get('expected_depth_ranges', []):
            print(f"    {c.get('class_name')}: {c.get('rule')} @ {r['top_m']}-{r['bottom_m']}m")

    # ====== Round 1: Code Execution ======
    print("\n--- Round 1: Code Execution (simulated) ---")
    code_results = MockDownstreamModels.traditional_code_execute(curves, raw_data)
    print(f"  Code returned {len(code_results)} segments")
    for r in code_results:
        print(f"    {r['id']}: {r['depth_top_m']}-{r['depth_bottom_m']}m ({r['rule']})")

    # ====== Round 1: Verification ======
    print("\n--- Round 1: VLM Verification ---")
    user_text = f"Plan: {json.dumps(curves, ensure_ascii=False)}\nResults: {json.dumps(code_results, ensure_ascii=False)}\n验证代码检测结果。仅输出JSON。"
    resp, t = call_vlm(verification_prompt, [log_img], user_text)
    verification = extract_json(resp)
    if not verification:
        print(f"  ❌ Verification failed")
        return plan
    print(f"  ✅ Verification complete ({t:.0f}s)")

    merges = verification.get('interval_merges', [])
    print(f"  Interval merges: {len(merges)}")
    for m in merges:
        print(f"    Merge {m.get('merge')}: {m.get('reason','')[:80]}")
    fluid_v = verification.get('fluid_verified', [])
    print(f"  Fluid verified: {len(fluid_v)}")
    for f in fluid_v:
        print(f"    {f.get('id')}: {f.get('actual_fluid')} (OK={f.get('fluid_ok')})")

    # ====== Final Output ======
    # 应用验证后的调整
    adjusted_segments = []
    merged_ids = set()
    for m in merges:
        for mid in m.get('merge', []):
            merged_ids.add(mid)

    for r in code_results:
        if r['id'] not in merged_ids:
            adjusted_segments.append(r)

    final = {
        "planning": plan,
        "verification": verification,
        "original_segments": len(code_results),
        "adjusted_segments": len(adjusted_segments),
        "merges_applied": len(merges),
    }
    print(f"\n  Final: {len(code_results)} original → {len(adjusted_segments)} adjusted "
          f"({len(merges)} merges)")
    return final


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LOOP TEST: SeismicInterpAgent (Plan→Detect→Verify)")
    print("=" * 60)

    seismic_img, seismic_data = generate_seismic(42)
    seismic_result = run_seismic_loop(seismic_img)

    print("\n\n" + "=" * 60)
    print("LOOP TEST: LogAnalysisAgent (Plan→Code→Verify→Merge)")
    print("=" * 60)

    log_img, log_data = generate_log(42)
    log_result = run_log_loop(log_img, log_data)

    # Save
    with open("/tmp/loop_test_results.json", 'w') as f:
        json.dump({"seismic": seismic_result, "log": log_result},
                  f, ensure_ascii=False, indent=2, default=str)
    print("\n\nSaved to /tmp/loop_test_results.json")
