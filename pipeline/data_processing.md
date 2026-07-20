# 数据处理流水线（Pipeline）

## 总览

```
原始数据                              VLM输入                     VLM输出
 ────────                             ────────                    ────────
 SEG-Y 3D地震体  ──预处理──▶  地震剖面图(PNG base64)  ──SeismicInterpAgent──▶  断层/层位/地震相 JSON
 LAS 测井文件    ──预处理──▶  测井曲线综合图(PNG)     ──LogAnalysisAgent──▶  岩性/物性/流体 JSON
 SEG-Y + LAS     ──预处理──▶  井震对比图(PNG) + 文本  ──WellSeismicFusionAgent──▶  井震融合 JSON
 前序JSON + 图   ──预处理──▶  汇总文本 + 异常图(PNG)  ──ProspectEvaluationAgent──▶  目标评价 JSON
```

## Agent 1: SeismicInterpAgent 数据处理

### 输入数据
- **SEG-Y 文件**：标准三维叠后地震数据体
- **可选**：地震属性体（相干、曲率、振幅包络）

### 预处理步骤

#### 1.1 数据验证
```python
import segyio
with segyio.open("seismic.segy", "r", strict=False) as f:
    print(f"Inline: {f.ilines.min()}-{f.ilines.max()}")
    print(f"Crossline: {f.xlines.min()}-{f.xlines.max()}")
    print(f"Samples: {f.samples.size}, Rate: {f.bin[segyio.BinField.Interval]/1000}ms")
```

#### 1.2 剖面提取
从3D体中间隔均匀抽取关键位置剖面：
- **Inline 剖面** (crossline × time)：均匀抽取 3-5 条
- **Crossline 剖面** (inline × time)：均匀抽取 3-5 条
- **时间切片** (inline × crossline)：关键目的层位置抽取 2-3 张

```python
# 核心逻辑
volume = segyio.tools.cube(f)
for il_idx in np.linspace(0, n_il-1, 5, dtype=int):
    section = volume[il_idx, :, :].T  # (time, xl)
    # → 绘图 → base64
```

#### 1.3 剖面可视化转图像

**显示规格**（对 VLM 识别效果影响大）：
- 色标：灰度 `cmap='gray'`（主分析），红蓝 `cmap='seismic'`（异常检测）
- 纵横比：`aspect='auto'`（压缩显示更贴合实际比例）
- 色阶范围：取 5%~95% 分位数，避免极端值压缩有效信号
- 分辨率：≥ 800×600 像素，保证断层细节可见
- 标注：坐标轴标注 inline/crossline 编号和时间(ms)

```python
import matplotlib.pyplot as plt
import io, base64

def section_to_image(section, cmap='gray', figsize=(12, 8), dpi=100):
    fig, ax = plt.subplots(figsize=figsize)
    vmin, vmax = np.percentile(section, [5, 95])
    ax.imshow(section, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xlabel('CDP / Crossline')
    ax.set_ylabel('Time (ms)')
    # → encode base64
```

#### 1.4 多图像组合发送
将 5-8 张关键剖面图一次性发给 VLM，让它综合判断：
```
[image: inline_100.png] [image: inline_300.png] [image: crossline_150.png]
请综合分析以上地震剖面，识别断层、层位...
```

### 输出解析

#### 1.5 JSON 提取与校验
VLM 输出可能包含额外的解释文字，需要：
1. 正则提取 `{...}` JSON 块：`re.search(r'\{[\s\S]*\}', response)`
2. 解析 JSON，逐字段类型校验
3. 坐标转换：VLM 识别的是图像像素坐标 → 映射回实际 inline/crossline/time

```python
# 像素坐标 → 实际坐标
x_inline = pixel_x / img_width * (il_max - il_min) + il_min
y_time = pixel_y / img_height * (t_max - t_min) + t_min
```

---

## Agent 2: LogAnalysisAgent 数据处理

### 输入数据
- **LAS 文件**：标准测井曲线数据（通常含 DEPTH, GR, SP, RT, AC, DEN, CNL 等）

### 预处理步骤

#### 2.1 LAS 读取与曲线标准化
```python
import lasio
las = lasio.read("well.las")
depth = las.depth_m  # 深度

# 曲线名称模糊匹配（LAS文件命名不规范）
curves = {}
for key in las.keys():
    kn = key.upper().strip()
    if 'GR' in kn or 'GAMMA' in kn: curves['GR'] = las[key].data
    elif 'SP' in kn: curves['SP'] = las[key].data
    elif any(c in kn for c in ['RT', 'RILD', 'ILD', 'RDEEP']): curves['RT'] = las[key].data
    elif any(c in kn for c in ['AC', 'DT', 'DTC']): curves['AC'] = las[key].data
    elif any(c in kn for c in ['DEN', 'RHOB']): curves['DEN'] = las[key].data
    elif any(c in kn for c in ['CNL', 'NPHI']): curves['CNL'] = las[key].data
```

#### 2.2 测井综合图绘制

**布局规范**（对 VLM 可读性至关重要）：
```
+-------+-------+----------+-----------+-----------+----------+
| GR    | SP    | 深电阻率 | AC        | DEN       | CNL      |
| 0-150 | -80+20| 0.1-1000 | 140-40    | 1.8-2.8   | 0.45--0.15|
| 绿    | 蓝    | 红(对数) | 橙        | 黑        | 品红     |
+-------+-------+----------+-----------+-----------+----------+
|                         深度 (m)                               |
+----------------------------------------------------------------+
```

关键要求：
- **6条标准曲线**：GR / SP / RT / AC / DEN / CNL
- 所有曲线共享纵轴（深度），从上到下（深度增加）
- 色阶范围固定以便跨井比较
- DPI ≥ 120，避免曲线细节模糊
- 井名和深度标尺必须清晰标注

#### 2.3 图像编码
```python
def log_to_image(curves, depth, well_name):
    fig, axes = plt.subplots(1, 6, figsize=(18, 14), sharey=True)
    # 逐一绘制 6 条曲线...
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    return base64.b64encode(buf.read()).decode()
```

### 输出解析

#### 2.4 VLM 输出处理
- 同上，正则提取 JSON
- 深度值校验：确保在各曲线有效深度范围内
- 岩性分类映射：将 VLM 输出的中文岩性名映射到标准分类编码

---

## Agent 3: WellSeismicFusionAgent 数据处理

### 输入数据
- **SEG-Y**：地震数据体
- **LAS**：测井曲线数据
- **Checkshot/时深对文件**：time(ms), depth(m) 对照表

### 预处理步骤

#### 3.1 井震标定（核心步骤）
```python
# 1. 从 checkshot 建立时深关系
td_pairs = [(800, 950), (1000, 1220), (1200, 1520), (1500, 2000)]  # (ms, m)
from scipy.interpolate import interp1d
depth_to_time = interp1d([d for _,d in td_pairs], [t for t,_ in td_pairs])

# 2. 将测井深度转为双程时间
log_time = depth_to_time(depth)

# 3. 提取井旁地震道（取井点位置 inline, xl 对应 trace）
well_trace = volume[il_idx, xl_idx, :]  # 井旁地震道
```

#### 3.2 井震对比图绘制

**布局**（从左到右）：
```
+-------------+------------------+-----------------------+
| 地震剖面    | 合成地震记录     | 测井曲线（GR+RT+AC） |
| (井旁CDP     | (反射系数+子波)  |                       |
|  集合)      |                  |                       |
+-------------+------------------+-----------------------+
                 深度/时间轴 (depth:time)
```

关键标注：
- 水平线标注关键地质界面（储层顶底）
- 标注井震相关系数
- 时深对应标注

#### 3.3 辅助文本输入
将以下信息以文本形式附加到 prompt：
```
井名: Well-A
井位: Inline 350, Crossline 180
时深关系: [(800ms,950m), (1000ms,1220m), (1200ms,1520m), (1500ms,2000m)]
测井可读取曲线: GR(0-150API), SP(-80～+20mV), RT(0.1-1000Ω·m), AC(140-40us/ft), DEN(1.8-2.8g/cm³), CNL(0.45～-0.15v/v)
```

### 输出解析
- 校验相关系数在 [0, 1] 范围
- 校验时深对单调性

---

## Agent 4: ProspectEvaluationAgent 数据处理

### 输入数据
- SeismicInterpAgent 输出的 JSON
- LogAnalysisAgent 输出的 JSON
- WellSeismicFusionAgent 输出的 JSON
- 可选：构造图、异常体平面展布图

### 预处理步骤

#### 4.1 前序结果汇总
将前三个 Agent 的 JSON 输出整理为结构化文本摘要：

```python
summary_text = f"""
## 地震解释结果
- 识别断层 {len(seismic_result['faults'])} 条
- 追踪层位 {len(seismic_result['horizons'])} 个
- 划分地震相 {len(seismic_result['seismic_facies'])} 种
- 构造圈闭 {len(seismic_result['structural_traps'])} 个
- 异常体 {len(seismic_result['anomalies'])} 个

断层详情: {json.dumps(seismic_result['faults'], ensure_ascii=False)}

## 测井分析结果
- 分析井数: {len(log_results)}
{format_log_summary(log_results)}

## 井震融合结果
- 井震相关系数: {fusion_result['well_seismic_calibration']['correlation_coefficient']}
- 关键地质界面: {len(fusion_result['key_geological_interfaces'])} 个
"""
```

#### 4.2 可选图像
- 圈闭平面分布图（inline × crossline 彩色标注）
- 异常体平面展布图
- 连井对比剖面

### 输出解析
- 风险评分 1-5 范围校验
- 地质成功率百分数校验
- 决策分类必须在 {drill_ready, data_gap, inventory, drop} 中

---

## 通用注意事项

1. **图像质量**：所有图像 DPI ≥ 100，尺寸适中（800-1500px），避免 VLM 放大后模糊
2. **坐标标注**：所有图像必须有清晰的坐标轴标注，VLM 才能输出有参考价值的坐标信息
3. **JSON 提取容错**：VLM 偶尔会在 JSON 外包裹 markdown 代码块，需要兼容处理
4. **重试策略**：若 JSON 解析失败，让 VLM 重新输出（temperature=0 减少随机性）
5. **批量处理**：多井/多剖面批量处理时，注意 GPU 显存和 token 限制
