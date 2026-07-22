"""
两阶段策略测试：VLM粗分 + 代码精确定位

Stage 1: VLM识别岩性/流体类型和大致深度范围
Stage 2: 传统代码用GR/RT阈值在numpy数据中精确定位边界
"""

import torch, json, io, re, warnings, numpy as np, time
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_PATH = "/data/yxjiang/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct"
print("Loading Qwen3-VL-8B..."); t0 = time.time()
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Loaded in {time.time()-t0:.0f}s")

# ====== 生成测井数据（保留原始numpy用于Stage 2精确提取） ======
depth = np.linspace(1000, 2000, 2000)
gr_raw = np.full(2000, 95.0); rt_raw = np.full(2000, 3.0)
ac_raw = np.full(2000, 80.0); den_raw = np.full(2000, 2.50)
cnl_raw = np.full(2000, 0.32); sp_raw = np.full(2000, 5.0)

for top, bot in [(1200,1255),(1400,1435),(1550,1625),(1750,1805)]:
    m=(depth>=top)&(depth<=bot); n=m.sum()
    gr_raw[m]=35+np.random.randn(n)*6; ac_raw[m]=63+np.random.randn(n)*4
    den_raw[m]=2.28+np.random.randn(n)*0.06; cnl_raw[m]=0.16+np.random.randn(n)*0.04
    sp_raw[m]=-35+np.random.randn(n)*6
m=(depth>=1555)&(depth<=1595); rt_raw[m]=85+np.random.randn(m.sum())*10
m=(depth>=1755)&(depth<=1790); rt_raw[m]=3+np.random.randn(m.sum())*0.3

# ====== Stage 1: VLM 粗分 ======
print("\n=== Stage 1: VLM粗分 ===")
curves = [('GR',gr_raw,'green','GR(API)',(0,150)),('SP',sp_raw,'blue','SP(mV)',(-60,20)),
    ('RT',rt_raw,'red','RT(Ohm.m)',(0.1,100)),('AC',ac_raw,'orange','AC(us/ft)',(140,40)),
    ('DEN',den_raw,'black','DEN(g/cm3)',(1.8,2.8)),('CNL',cnl_raw,'magenta','CNL(v/v)',(0.45,-0.15))]
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
buf.seek(0); log_img=Image.open(buf).convert('RGB')

# Stage 1 prompt: 不需要精确深度
stage1_prompt = """你是测井专家。分析测井曲线图，输出每个层段的大致岩性和流体类型。
只需输出JSON（精确深度由后续程序计算）:

{"zones":[{"depth_approx":"1200-1260","lithology":"sandstone","fluid":"water"},
          {"depth_approx":"1260-1400","lithology":"shale","fluid":null},
          {"depth_approx":"1550-1620","lithology":"sandstone","fluid":"gas"}],
 "summary":"识别出3套砂岩..."}"""

messages = [{"role":"system","content":stage1_prompt},
    {"role":"user","content":[{"type":"image","image":log_img},{"type":"text","text":"输出JSON。"}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=[log_img], return_tensors="pt").to(model.device)

t1=time.time()
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=4096, do_sample=True,
                            temperature=0.3, repetition_penalty=1.1, top_p=0.95)
vlm_time = time.time()-t1
resp = processor.decode(output[0], skip_special_tokens=True)
if "assistant" in resp: resp = resp.split("assistant")[-1].strip()

# Parse Stage 1 result
result = None
for m in re.finditer(r'\{', resp):
    start=m.start(); d=0; in_s=False; esc=False; end=-1
    for i in range(start, len(resp)):
        c=resp[i]
        if esc: esc=False; continue
        if c=='\\': esc=True; continue
        if c=='"' and not esc: in_s=not in_s; continue
        if in_s: continue
        if c=='{': d+=1
        elif c=='}': d-=1
        if d==0: end=i+1; break
    if end>start:
        try:
            r=json.loads(resp[start:end])
            if isinstance(r,dict) and 'zones' in r: result=r; break
        except: continue

print(f"VLM time: {vlm_time:.0f}s | JSON: {'✅' if result else '❌'}")
if result:
    for z in result.get('zones',[]):
        print(f"  VLM: {z.get('depth_approx','?')} | {z.get('lithology','?')} | fluid={z.get('fluid','?')}")

# ====== Stage 2: 代码精确定位 ======
print("\n=== Stage 2: 代码精确定位 ===")

def find_boundaries(curve, depth, threshold, direction='below'):
    crossings = []
    for i in range(1, len(curve)):
        if direction == 'below':
            if curve[i-1] >= threshold and curve[i] < threshold:
                crossings.append(float(depth[i]))
            elif curve[i-1] < threshold and curve[i] >= threshold:
                crossings.append(float(depth[i]))
        else:
            if curve[i-1] <= threshold and curve[i] > threshold:
                crossings.append(float(depth[i]))
            elif curve[i-1] > threshold and curve[i] <= threshold:
                crossings.append(float(depth[i]))
    return crossings

# 用 GR<50 找砂岩边界
gr_boundaries = find_boundaries(gr_raw, depth, 50, 'below')
print(f"GR<50 crossings: {len(gr_boundaries)} points")
for b in gr_boundaries[:12]:
    print(f"  {b:.1f}m")

# 用 RT>20 找含气段
rt_boundaries = find_boundaries(rt_raw, depth, 20, 'above')
print(f"RT>20 crossings: {len(rt_boundaries)} points")
for b in rt_boundaries[:8]:
    print(f"  {b:.1f}m")

# ====== Stage 3: 合并 ======
print("\n=== 合并结果 ===")

# 代码精确提取砂岩段
sand_intervals = []
in_sand = False; sand_start = 0
for i in range(1, len(gr_raw)):
    if not in_sand and gr_raw[i] < 50 and gr_raw[i-1] >= 50:
        sand_start = depth[i]; in_sand = True
    elif in_sand and gr_raw[i] >= 50 and gr_raw[i-1] < 50:
        sand_intervals.append((float(sand_start), float(depth[i]))); in_sand = False
if in_sand:
    sand_intervals.append((float(sand_start), float(depth[-1])))

# 过滤噪音（厚度>10m才算有效砂层）
sand_intervals = [(t,b) for t,b in sand_intervals if b-t > 10]

# 每个砂层的流体类型
final_result = {"zones": []}
for top, bot in sand_intervals:
    fluid = "unknown"
    m = (depth >= top) & (depth <= bot)
    avg_rt = rt_raw[m].mean()
    avg_den = den_raw[m].mean()
    if avg_rt > 20 and avg_den < 2.35:
        fluid = "gas"
    elif avg_rt > 20:
        fluid = "oil"
    elif avg_rt < 5:
        fluid = "water"
    final_result["zones"].append({
        "depth_top": round(top, 1), "depth_bottom": round(bot, 1),
        "lithology": "sandstone", "fluid": fluid,
        "rt_avg": round(avg_rt, 1), "source": "code_precise",
    })

# GT comparison
gt_sands = [(1200,1255),(1400,1435),(1550,1625),(1750,1805)]
gt_fluids = {1555:'gas', 1755:'water'}

print("\nFinal output (code-precise):")
for z in final_result['zones']:
    print(f"  {z['depth_top']}-{z['depth_bottom']}m: sandstone | {z['fluid']} (RT={z['rt_avg']})")

print(f"\nGround truth sands: {gt_sands}")
print(f"Ground truth fluids: gas@1555-1595, water@1755-1790")

# Precision matching
matches = 0
for z in final_result['zones']:
    for gt_t, gt_b in gt_sands:
        if abs(z['depth_top'] - gt_t) < 30 and abs(z['depth_bottom'] - gt_b) < 30:
            matches += 1; break
print(f"\nPrecision sand match (depth±30m): {matches}/{len(gt_sands)}")

# Recall matching
hit_count = 0
for gt_t, gt_b in gt_sands:
    for z in final_result['zones']:
        if z['depth_top'] <= gt_b and z['depth_bottom'] >= gt_t:
            hit_count += 1; break
print(f"Sand recall (any overlap): {hit_count}/{len(gt_sands)}")

# Fluid accuracy
fluid_correct = 0
for z in final_result['zones']:
    for gt_depth, gt_fluid in gt_fluids.items():
        if z['depth_top'] <= gt_depth <= z['depth_bottom'] and z['fluid'] == gt_fluid:
            fluid_correct += 1
print(f"Fluid accuracy: {fluid_correct}/{len(gt_fluids)} (code-based, 100% dependent on thresholds)")

print(f"\n{'='*60}")
print(f"Two-stage: VLM {vlm_time:.0f}s + code <0.01s = {vlm_time:.0f}s total")
print(f"Depth precision: from code (GR threshold), not VLM guesswork")
print(f"Fluid identification: from code (RT+DEN threshold), not VLM guesswork")
