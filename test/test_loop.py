"""
闭环Agent工作流 — VLM自主决定调用哪些下游模型

VLM = 大脑：看图像 → 思考需要什么 → 自主选择下游模型 → 验证结果 → 迭代
"""

import torch, json, io, re, warnings, numpy as np, time
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MAX_ITERATIONS = 3
MODEL_PATH = "/data/yxjiang/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct"

print("Loading VLM...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)


def call_vlm(system_prompt, images, user_text, max_tokens=4096, temperature=0.3):
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": user_text})
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if images:
        inputs = processor(text=text, images=images, return_tensors="pt").to(model.device)
    else:
        inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                                temperature=temperature, repetition_penalty=1.1, top_p=0.95)
    resp = processor.decode(output[0], skip_special_tokens=True)
    if "assistant" in resp: resp = resp.split("assistant")[-1].strip()
    return resp, time.time() - t0


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
                if isinstance(r, dict) and len(r) >= 2: return r
            except: continue
    return None


# ============================================================
# 下游模型注册表 — VLM可以从中选择
# ============================================================

AVAILABLE_MODELS = """
可用的下游模型:
1. yolo_world: 开放词汇目标检测，输入class_name+expected_range → 输出bbox列表
   适合: 断层、亮点异常、河道、盐丘等离散目标的检测
2. sam: Segment Anything Model，输入point或bbox prompts → 输出像素级mask
   适合: 层位追踪、盐体分割、异常体轮廓精确勾画
3. traditional_code: 阈值规则执行引擎，输入rule+expected_range → 输出精确数值(±0.1m)
   适合: 测井曲线分析(GR<50=砂岩)、振幅提取、断层属性计算(coherence/curvature)
4. segformer: 语义分割，输入类别描述 → 输出全图分割mask
   适合: 地震相分类、岩性剖面预测
"""

AVAILABLE_DOWNSTREAM = {
    "yolo_world": {
        "description": "开放词汇目标检测",
        "required_fields": ["class_name", "description", "expected_range", "confidence_threshold"],
        "output": "bbox列表 [{id, class_name, bbox: [x1,y1,x2,y2], confidence}]",
    },
    "sam": {
        "description": "Segment Anything Model",
        "required_fields": ["prompt_type(point/bbox)", "prompt_value", "label"],
        "output": "mask (像素级分割)",
    },
    "traditional_code": {
        "description": "阈值规则执行引擎",
        "required_fields": ["class_name", "rule", "expected_range"],
        "output": "精确数值区间 [{id, depth_top, depth_bottom, value}]",
    },
}

# ============================================================
# 模拟下游模型执行
# ============================================================

def execute_downstream(model_name: str, instruction: dict, context: dict) -> list:
    """
    下游模型路由。VLM选择了model_name，这里执行它。
    真实部署时替换为实际API调用。
    """
    if model_name == "yolo_world":
        # 真实: model.detect(image, instruction['class_name'])
        # 模拟: 在expected_range内随机生成bbox
        results = []
        for cat in instruction.get('categories', [instruction]):
            cdp = cat.get('expected_cdp_range', [0, 300])
            t = cat.get('expected_time_range_ms', [0, 2500])
            conf = np.random.uniform(0.4, 0.9)
            results.append({
                "id": f"yolo_{cat.get('class_name','det')[:8]}_{np.random.randint(100)}",
                "class_name": cat.get('class_name'),
                "bbox": [cdp[0]+5, cdp[1]-5, t[0]+30, t[1]-30],
                "confidence": round(conf, 2),
                "model": "yolo_world",
            })
        return results

    elif model_name == "sam":
        label = instruction.get('label', 'segment')
        return [{
            "id": f"sam_{label}_{np.random.randint(100)}",
            "label": label,
            "mask_area_pixels": np.random.randint(5000, 50000),
            "model": "sam",
        }]

    elif model_name == "traditional_code":
        results = []
        # VLM可能传不同格式: rules[] 或直接字段
        if 'rules' in instruction:
            rules = instruction['rules']
        elif 'rule' in instruction:
            rules = [instruction]
        else:
            # VLM传了自定义指令（如振幅计算），模拟返回
            return [{
                "id": f"code_custom_{np.random.randint(100)}",
                "class_name": instruction.get('class_name', 'custom'),
                "result": "computed",
                "instruction": instruction,
                "model": "traditional_code",
            }]

        for rule in rules:
            # VLM可能输出各种格式，先适配
            if isinstance(rule, str):
                results.append({"id": f"code_str_{np.random.randint(100)}", "class_name": rule, "model": "traditional_code"})
                continue
            ranges = rule.get('expected_depth_ranges', [])
            if not ranges:
                # 可能是其他格式: expected_range / target_range / ...
                ranges = (rule.get('expected_range', []) or rule.get('target_range', [])
                          or rule.get('depth_range', []))
            if not ranges:
                results.append({
                    "id": f"code_{rule.get('class_name','seg')}_{np.random.randint(100)}",
                    "class_name": rule.get('class_name'),
                    "rule": rule.get('rule', str(rule)[:50]),
                    "model": "traditional_code",
                })
                continue
            for r in ranges:
                if isinstance(r, dict):
                    top = r.get('top_m', r.get('top', 0)) + np.random.randn() * 3
                    bot = r.get('bottom_m', r.get('bottom', 0)) + np.random.randn() * 3
                elif isinstance(r, (list, tuple)) and len(r) >= 2:
                    top = r[0] + np.random.randn() * 3
                    bot = r[1] + np.random.randn() * 3
                else:
                    continue
                if bot - top > 30:
                    mid = (top + bot) / 2
                    results.append({
                        "id": f"code_{rule.get('class_name','seg')}_{np.random.randint(100)}",
                        "class_name": rule.get('class_name'),
                        "depth_top_m": round(top, 1),
                        "depth_bottom_m": round(mid - 1 + np.random.randn(), 1),
                        "rule": rule.get('rule', ''),
                        "model": "traditional_code",
                    })
                    results.append({
                        "id": f"code_{rule.get('class_name','seg')}_{np.random.randint(100)}",
                        "class_name": rule.get('class_name'),
                        "depth_top_m": round(mid + 1 + np.random.randn(), 1),
                        "depth_bottom_m": round(bot, 1),
                        "rule": rule.get('rule', ''),
                        "model": "traditional_code",
                    })
                else:
                    results.append({
                        "id": f"code_{rule.get('class_name','seg')}_{np.random.randint(100)}",
                        "class_name": rule.get('class_name'),
                        "depth_top_m": round(top, 1),
                        "depth_bottom_m": round(bot, 1),
                        "rule": rule.get('rule', ''),
                        "model": "traditional_code",
                    })
        return results

    return []


# ============================================================
# VLM 自主决策 Workflow
# ============================================================

WORKFLOW_PLANNING_PROMPT = f"""你是地球物理AI工作流规划器。分析图像后，自主决定需要调用哪些下游模型、按什么顺序、用哪些参数。

{AVAILABLE_MODELS}

输出一个完整的工作流计划:
{{
  "scene_understanding": "对图像内容的理解摘要",
  "workflow_steps": [
    {{
      "step": 1,
      "model": "yolo_world|sam|traditional_code|segformer",
      "reason": "为什么选择这个模型（例如：需要检测离散的断层目标→选yolo_world；需要精确定位岩性边界→选traditional_code）",
      "instruction": {{
        // 根据所选模型填写对应的required_fields
        // yolo_world: categories[{{class_name, description, expected_cdp_range, expected_time_range_ms, confidence_threshold}}]
        // sam: {{prompt_type, prompt_value, label}}
        // traditional_code: rules[{{class_name, rule, expected_depth_ranges}}]
      }}
    }}
  ],
  "dependencies": [{{"step": 3, "depends_on": [1, 2]}}],
  "verification_strategy": "per_step|batch|none",
  "max_iterations": 2
}}

仅输出JSON。"""


VERIFICATION_PROMPT = """你是地球物理验证专家。根据原始图像验证下游模型的每条检测结果。

原始工作流: {workflow_plan}
下游执行结果: {downstream_results}

逐条验证:
1. 这条检测是否真实地质特征？（对照原始图像判断）
2. 如果是假阳性，原因是什么？
3. 有没有遗漏的目标（图像可见但模型没检测到）？
4. 是否需要追加步骤或调整参数重新检测？

JSON:
{"verified":[{"step":1,"model":"yolo_world","result_id":"...","is_real":true/false,
   "confidence":0.9,"geological_reason":"...","rejection_reason":null}],
 "false_positives":2,
 "missed_targets":[{"suggested_model":"yolo_world","class_name":"...","expected_range":{...},"reason":"..."}],
 "need_retry":true/false,
 "retry_instructions":{"step":1,"model":"...","adjusted_params":{...},"reason":"..."},
 "final_summary":"验证总结"}"""


# ============================================================
# 完整闭环
# ============================================================

def run_agent_loop(image, agent_name="agent"):
    """VLM自主规划→执行→验证→重试 的完整闭环"""

    all_results = []

    # ====== Phase 1: VLM 自主规划 workflow ======
    print(f"\n{'='*60}")
    print(f"Phase 1: VLM 自主分析 + 选择下游模型")
    print(f"{'='*60}")

    resp, t = call_vlm(WORKFLOW_PLANNING_PROMPT, [image],
                       "分析图像，输出完整workflow计划。仅输出JSON。")
    plan = extract_json(resp)
    if not plan:
        print(f"  ❌ Planning failed. Raw: {resp[:400]}")
        return None

    print(f"  ✅ Plan generated ({t:.0f}s)")
    print(f"  Scene: {plan.get('scene_understanding', '')[:150]}")
    steps = plan.get('workflow_steps', [])
    print(f"  Steps: {len(steps)}")
    for s in steps:
        print(f"    Step{s.get('step')}: {s.get('model')} → {s.get('reason','')[:80]}")

    # ====== Phase 2: 执行 workflow（按VLM决定的步骤） ======
    print(f"\nPhase 2: 执行下游模型（按VLM规划的顺序）")
    print(f"{'='*60}")

    for iteration in range(plan.get('max_iterations', 2)):
        round_results = []

        for step in steps:
            model_name = step.get('model')
            instruction = step.get('instruction', {})
            step_num = step.get('step')

            if model_name not in AVAILABLE_DOWNSTREAM:
                print(f"  ⚠️ Step{step_num}: unknown model '{model_name}', skip")
                continue

            print(f"  Step{step_num}: calling {model_name}...")
            results = execute_downstream(model_name, instruction, {})
            round_results.extend(results)
            print(f"    → {len(results)} results")
            for r in results:
                info = str(r)[:120]
                print(f"      {info}")

        if not round_results:
            print("  No results, skip verification")
            break

        all_results.extend(round_results)

        # ====== Phase 3: VLM 验证 ======
        print(f"\nPhase 3: VLM 验证结果 (iteration {iteration+1})")
        print(f"{'='*60}")

        user_text = (f"Workflow: {json.dumps(steps, ensure_ascii=False)}\n"
                     f"Results: {json.dumps(round_results, ensure_ascii=False)}\n"
                     f"逐条验证检测结果。仅输出JSON。")
        resp, t = call_vlm(VERIFICATION_PROMPT, [image], user_text)
        ver = extract_json(resp)

        if not ver:
            print(f"  ❌ Verification failed")
            break

        print(f"  ✅ Verified ({t:.0f}s)")
        verified = ver.get('verified', [])
        real = sum(1 for v in verified if v.get('is_real'))
        fp = sum(1 for v in verified if not v.get('is_real'))
        print(f"  Real: {real} | False positive: {fp} | Need retry: {ver.get('need_retry')}")

        if not ver.get('need_retry'):
            break

        # Retry: 根据VLM建议调整参数
        retry = ver.get('retry_instructions', {})
        print(f"  Retry: {retry.get('reason', '')[:100]}")
        # 更新对应step的参数
        for s in steps:
            if s.get('step') == retry.get('step'):
                s['instruction'] = retry.get('adjusted_params', s.get('instruction'))

    # ====== Final ======
    final = {
        "agent": agent_name,
        "plan": plan,
        "results": all_results,
        "verification": ver if 'ver' in dir() else None,
    }
    return final


# ============================================================
# 数据生成
# ============================================================

def generate_seismic(seed=42):
    np.random.seed(seed)
    n_traces, n_samples = 300, 500
    reflectivity = np.zeros((n_samples, n_traces))
    for layer_idx in range(8):
        base_depth = 50 + layer_idx * 55
        for i in range(n_traces):
            depth = base_depth + (15+np.random.rand()*20) * np.sin(i/(80+np.random.rand()*40)*2*np.pi)
            d_idx = int(np.clip(depth+np.random.randn()*3, 0, n_samples-1))
            if d_idx < n_samples:
                reflectivity[d_idx, i] = np.random.rand()*0.6 + 0.2
    reflectivity[:, 120:] = np.roll(reflectivity[:, 120:], 15, axis=0)
    reflectivity[240:243, 180:210] = -1.5
    for i in range(50, 80):
        thickness = int(8*(1-((i-65)/15)**2))
        if thickness > 0:
            reflectivity[300:300+thickness, i] = 0.4+np.random.randn(thickness)*0.15

    def ricker(l, dt, f):
        t = np.arange(l)*dt-(l*dt)/2
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
    return Image.open(buf).convert('RGB')


def generate_log(seed=42):
    np.random.seed(seed)
    depth = np.linspace(1000, 2000, 2000)
    gr = np.full(2000, 95.0); rt = np.full(2000, 3.0)
    den = np.full(2000, 2.50); cnl = np.full(2000, 0.32)
    for top, bot in [(1200,1255),(1400,1435),(1550,1625),(1750,1805)]:
        m=(depth>=top)&(depth<=bot); n=m.sum()
        gr[m]=35+np.random.randn(n)*6
        den[m]=2.28+np.random.randn(n)*0.06; cnl[m]=0.16+np.random.randn(n)*0.04
    m=(depth>=1555)&(depth<=1595); rt[m]=85+np.random.randn(m.sum())*10

    curves = [('GR',gr,'green','GR(API)',(0,150)),('RT',rt,'red','RT(Ohm.m)',(0.1,100)),
              ('DEN',den,'black','DEN(g/cm3)',(1.8,2.8)),('CNL',cnl,'magenta','CNL(v/v)',(0.45,-0.15))]
    fig,axes=plt.subplots(1,4,figsize=(12,14),sharey=True)
    for ax,(name,data,color,label,xlim) in zip(axes,curves):
        ax.plot(data,depth,color=color,linewidth=0.5)
        ax.set_xlabel(label,fontsize=8); ax.set_xlim(xlim)
        ax.grid(True,alpha=0.3); ax.invert_yaxis()
        if xlim[0]>xlim[1]: ax.invert_xaxis()
    axes[0].set_ylabel('Depth(m)')
    fig.suptitle('Well Log',fontsize=12,fontweight='bold')
    plt.tight_layout()
    buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=120); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AUTONOMOUS AGENT: Seismic Section")
    print("=" * 60)
    seismic_img = generate_seismic(42)
    seismic_result = run_agent_loop(seismic_img, "seismic")

    print("\n\n" + "=" * 60)
    print("AUTONOMOUS AGENT: Well Log")
    print("=" * 60)
    log_img = generate_log(42)
    log_result = run_agent_loop(log_img, "log")

    # Summary
    print("\n\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for name, result in [("seismic", seismic_result), ("log", log_result)]:
        if not result:
            print(f"  {name}: FAILED")
            continue
        plan = result.get('plan', {})
        steps = plan.get('workflow_steps', [])
        models_used = [s.get('model') for s in steps]
        ver = result.get('verification', {})
        print(f"  {name}:")
        print(f"    Scene: {plan.get('scene_understanding','')[:120]}")
        print(f"    Models chosen: {models_used}")
        print(f"    Steps: {len(steps)} | Results: {len(result.get('results',[]))}")

    with open("/tmp/autonomous_agent_results.json", 'w') as f:
        json.dump({"seismic": seismic_result, "log": log_result},
                  f, ensure_ascii=False, indent=2, default=str)
    print("\nSaved to /tmp/autonomous_agent_results.json")
