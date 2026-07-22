# GeoMultimodal Input Adapter

地球物理多模态输入适配接口。项目把地震、测井、井位、井轨迹及时深数据转换为标准图像、数值数组、Mask、质量报告、动态 Prompt 和模型无关的 `request.json`，供 Qwen3-VL 等多模态模型使用。

本项目只生成数据适配结果和模型请求包，不部署或调用大模型，不实现 YOLO-World，也不会在依据不足时声称识别出真实油气藏。

> `examples/data/` 中的全部数据仅用于测试软件流程，不代表真实地质规律，不包含真实井、真实坐标或真实油气藏标签。

## 1. 系统位置

```text
原始 SEG-Y / NumPy / LAS / CSV / JSON / YAML
  → GeoMultimodal Input Adapter
  → 图像 + 数组/Mask + manifest + QC + Prompt + request
  → Qwen3-VL 等多模态模型
  → 结构化 JSON
  → YOLO-World 等下游检测/分割流程
```

Qwen3-VL 不直接读取 SEG-Y 或 LAS。本项目先生成独立物理视图和结构化上下文。inline、crossline、时间/深度切片不会被拼成伪 RGB 三通道。

## 2. 已实现能力

- 地震：`.npy`、`.npz`，以及安装 `segyio` 后的 `.sgy`、`.segy`。
- 测井：CSV，以及安装 `lasio` 后的 LAS。
- 井位/轨迹/时深表：CSV、JSON、YAML。
- 精确别名优先的九槽位曲线映射，支持中英文字段和配置覆盖。
- 单位转换、短缺口插值、原始有效 Mask、插值 Mask、整条曲线缺失状态。
- 三类电阻率的探测深度和测量体系分别保存；候选不平均。
- 最小曲率轨迹、TVDSS 转换、绝对坐标与偏移一致性检查。
- AC/DT 积分、控制点仿射标定、Checkshot/VSP/外部时深表读取。
- H0–H3、V0–V4 等级和五级融合权限自动判断。
- 地震模型图/QC 图、四轨测井综合图及单槽位图。
- Pydantic manifest、JSON Schema、动态 Prompt、模型无关 request。
- CLI、Python API、输入检查、运行包完整性校验和可复现记录。
- SEG-Y 只扫描几何并按需读取 inline/crossline，不无条件把三维体全部载入内存。

## 3. 九类测井语义

| 槽位 | 物理量 | 典型原始名 | 内部单位 |
|---|---|---|---|
| `GR` | 自然伽马 | GR、GAMMA | API |
| `SP` | 自然电位 | SP | mV |
| `CAL` | 井径 | CAL、CALI | inch |
| `RES_DEEP` | 深电阻率 | ILD、LLD、RT | ohm_m |
| `RES_MEDIUM_SHALLOW` | 中/浅电阻率 | ILM、LLS | ohm_m |
| `RES_MICRO` | 微电阻率 | MSFL、MLL、RXO | ohm_m |
| `AC` | 声波时差 | AC、DT、DTC | us/m |
| `DEN` | 体积密度 | DEN、RHOB | g/cm3 |
| `CNL` | 中子孔隙度 | CNL、NPHI | fraction |

固定的是内部物理语义，不是原始 mnemonic。缺失 SP 不会用 GR 补；微电阻率不会替代浅电阻率。别名位于 `configs/curve_aliases.yaml`，用户可通过 `preferred_curves` 和 `resistivity_overrides` 显式覆盖选择。

### 电阻率规则

电阻率同时保存两个维度：

- 探测深度：`deep / medium / shallow / micro / unknown`。
- 测量体系：`induction / laterolog / micro_focused / other / unknown`。

例如 ILD 和 LLD 都可以是深电阻率候选，但分别保留感应与侧向测量体系。程序选一个首选曲线并保留候选详情，不会求平均。非正值在 `log10` 前标记无效；清洗表额外保存 `*_LOG10` 列。

## 4. 深度、高程和坐标

- `MD`：沿井眼累计测量井深。
- `TVD`：从已声明深度参考面向下的真实垂直距离。
- `TVDSS`：相对平均海平面的真实垂直深度；默认内部符号为海平面以下为正。
- `KB`：Kelly Bushing 高程，也可能是测井深度起算面；代码不会把它与地面海拔混用。
- 地面海拔：地面相对垂向基准面的高程。
- 完钻井深：必须另行说明属于 MD、TVD 或其他类型；未提供类型时保持 `unknown`。

只有在 TVD 参考面、KB 高程基准、符号和单位明确时，才使用 `TVDSS = TVD - KB_elevation`。只有 MD/TVD 时只能进行垂向转换；无 INC/AZI 时不会恢复地下 X/Y；只给井口坐标且未声明直井时，不会默认整口井为直井。

CRS 建议使用明确 EPSG 或完整 PROJ 定义，例如 `EPSG:32631`。仅写 `WGS84`、`Beijing1954`、`CGCS2000` 但没有投影/分带参数时会标为 `ambiguous`。井和地震 CRS 都明确且一致，或通过 `pyproj` 完成可记录的转换后，才可能达到 H3。程序不会根据坐标数值“看起来接近”推断 CRS。

## 5. AC/DT 时深关系

慢度统一为 `us/m` 后，按确认的垂向轴积分：

```text
TWT(z) = t0 + 2 × integral(slowness(z) dz)
```

优先使用 TVDSS；斜井不能把 MD 积分直接当垂向地震时间。短缺口可配置插值，长缺口不跨越积分。支持：

- Checkshot/VSP/外部时深表；
- AC/DT 未标定积分；
- 使用两个以上覆盖范围内控制点做仿射校正，并报告 RMSE 与相关系数；
- 透明的基础一维合成地震记录函数，默认不执行自动井震相关匹配。

项目没有统一 `t0` 或统一替换速度默认值，因为这些参数取决于井级基准、浅层速度、采集与处理口径。未提供 t0、井级替换速度或控制点时，积分结果只相对声波起测点；未标定 AC/DT 只能作为低可信度粗略参考。

## 6. 配准和融合权限

水平等级：

| 内部值 | 含义 |
|---|---|
| `none` | H0，无井位 |
| `wellhead_only` | H1，只有井口位置 |
| `trajectory_available` | H2，有地下轨迹，但未确认与地震同一 CRS |
| `seismic_crs_aligned` | H3，地下轨迹和地震坐标系已明确统一 |

垂向等级：

| 内部值 | 含义 |
|---|---|
| `none` | V0，无垂向关系 |
| `depth_reference_only` | V1，只有统一深度基准 |
| `sonic_uncalibrated` | V2，AC/DT 积分未精细标定 |
| `sonic_calibrated` | V3，AC/DT 已做井级控制点标定 |
| `measured_time_depth` | V4，Checkshot/VSP 或外部时深关系 |

融合权限从低到高为 `separate_analysis_only`、`location_level_association`、`approximate_vertical_mapping`、`calibrated_joint_analysis`、`precise_joint_analysis`。例如 H3+V2 只允许粗略纵向映射；CRS 不明确时不会得到 H3。

## 7. 缺失数据策略

缺失从不等同于零。每条可用曲线均保存：

- `available`；
- 原始 `valid_mask`；
- `interpolated_mask`；
- `missing_ratio` 和最大连续缺口；
- `warnings`、`limitations` 和处理记录。

固定 `(N, 9)` 数组中，缺失或长缺口位置使用零占位，但必须与两个 Mask 及 `curve_available.npy` 一起读取。清洗 CSV 保持 NaN，不用零伪装有效测量。整个地震或测井模态缺失时，可分别进入 `seismic_only` 或 `well_log_only`；两者都缺失则运行无效。

## 8. 安装

要求 Python 3.10 及以上。在项目根目录执行：

```powershell
python -m pip install -e .
```

按输入格式安装可选能力：

```powershell
python -m pip install -e ".[las]"      # LAS
python -m pip install -e ".[segy]"     # SEG-Y
python -m pip install -e ".[crs]"      # CRS 转换
python -m pip install -e ".[all,test]" # 全部可选能力与测试
```

如果请求某种格式但缺少对应依赖，CLI 会返回清楚的安装提示，不会输出难以理解的导入异常。

## 9. 端到端操作流程

本项目的标准工作流如下：

```text
准备原始文件
  → 创建样本配置 YAML
  → inspect 检查字段、单位和风险
  → prepare 生成标准模型输入包
  → validate 验证包内引用和配准权限
  → 模型调用端读取 request.json
  → Qwen3-VL 接收 PNG 与文本
  → 校验并保存模型输出 JSON
```

### 9.1 首次安装

在 PowerShell 中进入项目根目录：

```powershell
cd "C:\TT\竞赛\揭榜挂帅\多模态接口"
python -m pip install -e .
```

根据实际输入格式安装可选依赖：

```powershell
python -m pip install -e ".[las]"   # 读取 LAS
python -m pip install -e ".[segy]"  # 读取 SEG-Y
python -m pip install -e ".[crs]"   # 使用 pyproj 转换 CRS
```

### 9.2 运行内置示例

仓库已经包含生成后的合成示例。需要重新生成时执行：

```powershell
python examples/generate_sample_data.py
```

执行完整流程：

```powershell
geo-adapter inspect --config examples/sample_config.yaml
geo-adapter prepare --config examples/sample_config.yaml
geo-adapter validate --run-dir runs/demo_sample_001
geo-adapter show-manifest --run-dir runs/demo_sample_001
```

如果终端无法识别 `geo-adapter`，使用等价模块命令：

```powershell
python -m geo_adapter.cli inspect --config examples/sample_config.yaml
python -m geo_adapter.cli prepare --config examples/sample_config.yaml
python -m geo_adapter.cli validate --run-dir runs/demo_sample_001
python -m geo_adapter.cli show-manifest --run-dir runs/demo_sample_001
```

命令职责：

- `inspect`：只读检查输入，不生成完整运行包。
- `prepare`：执行读取、映射、清洗、配准分级、制图和打包。
- `validate`：检查 Schema、相对路径、PNG、数组/Mask 维度和融合权限。
- `show-manifest`：显示运行模式、可用模态、H/V 等级和质量状态。

CLI 失败时返回非零退出码。`output.overwrite: true` 会在再次运行时重建对应运行目录；需要保留人工修改结果时，应先复制原运行目录。

## 10. 原始数据接入

### 10.1 推荐的数据组织方式

原始数据可位于任意磁盘目录。为便于管理，推荐按样本建立独立目录：

```text
data/
└── my_sample_001/
    ├── seismic.sgy
    ├── well_log.las
    ├── well_location.csv
    ├── trajectory.csv
    └── time_depth.csv
```

五类输入不要求全部存在。缺失模态在配置中使用 `path: null` 并设置 `optional: true`，不得创建伪造文件补齐目录。

### 10.2 地震输入

支持 `.sgy`、`.segy`、`.npy`、`.npz`：

```yaml
inputs:
  seismic:
    path: "C:/data/my_sample_001/seismic.sgy"
    format: auto
    domain: time
    crs: "EPSG:32650"
    optional: false
```

配置要求：

- `domain` 应使用 `time`、`depth` 或 `unknown`。只有在文件元数据可靠时才使用 `auto`。
- `crs` 应填写明确 EPSG 或完整 CRS 参数；未知时使用 `null`。
- SEG-Y 必须安装 `.[segy]`。
- 大型 SEG-Y 只扫描几何并按需提取视图，不无条件加载整个三维体。

NumPy 三维数组默认轴顺序为 `[inline, crossline, sample]`。NPZ 首选数组键为 `amplitude`，也可通过 `array_key` 指定：

```yaml
inputs:
  seismic:
    path: "C:/data/my_sample_001/seismic.npz"
    array_key: amplitude
    domain: time
```

二维 NumPy 数组按用户提供的物理 Patch 处理，不会自动标记为 inline。

### 10.3 测井输入

LAS 可直接使用。程序读取 LAS 曲线 mnemonic、描述和单位：

```yaml
inputs:
  well_log:
    path: "C:/data/my_sample_001/well_log.las"
    format: auto
    well_id: "WELL_001"
    optional: false
```

CSV 至少包含一列深度和一列测井值。推荐格式：

```csv
MD,GR,SP,CALI,ILD,ILM,MSFL,DTC,RHOB,NPHI
1000.0,65.2,-20.1,8.5,12.3,8.2,3.1,302.5,2.35,0.22
1000.5,66.1,-19.8,8.6,12.8,8.4,3.2,301.9,2.36,0.23
```

CSV 通常不携带单位，必须在配置中明确：

```yaml
processing:
  well_logs:
    curve_units:
      GR: API
      SP: mV
      CALI: inch
      ILD: ohm_m
      ILM: ohm_m
      MSFL: ohm_m
      DTC: us/m
      RHOB: g/cm3
      NPHI: fraction
```

`NPHI/CNL` 的配置规则：

- 数值 `0.225` 表示 22.5% 时，单位填写 `fraction`。
- 数值 `22.5` 表示 22.5% 时，单位填写 `%`。
- 单位无法确认时留空，程序保留原值并产生警告，不静默换算。

新增厂商曲线名时，在 `configs/curve_aliases.yaml` 中增加精确别名。多条曲线命中同一物理槽位时，可指定首选曲线：

```yaml
processing:
  well_logs:
    preferred_curves:
      RES_DEEP: ILD
```

程序保留全部候选及其测量体系，不对感应和侧向电阻率求平均。

### 10.4 井位输入

推荐 CSV：

```csv
WELL,X,Y,KB,GL,TD,CRS
WELL_001,500000,6500000,75,69,3200,EPSG:32650
```

字段含义：

- `WELL`：井名。
- `X/Y`：井口坐标。
- `KB`：补心高程。
- `GL`：地面海拔。
- `TD`：完钻井深；未提供类型时保持 `unknown`。
- `CRS`：明确的坐标参考系。

经纬度输入示例：

```csv
WELL,LONGITUDE,LATITUDE
WELL_001,120.1234,30.5678
```

明确的经纬度字段按 WGS84 经纬度处理。仅有 `WGS84`、`Beijing1954`、`CGCS2000` 等名称但缺少投影参数时，CRS 状态为 `ambiguous`。

### 10.5 井轨迹输入

完整轨迹示例：

```csv
WELL,MD,TVD,TVDSS,X,Y
WELL_001,1000,980,905,500010,6500005
WELL_001,1100,1075,1000,500020,6500012
```

测斜数据示例：

```csv
WELL,MD,INC,AZI
WELL_001,1000,2.0,45.0
WELL_001,1100,5.0,48.0
```

处理规则：

- `MD + INC + AZI`：使用最小曲率法计算 TVD 和 X/Y 偏移。
- `MD + TVD`：只进行垂向转换，地下 X/Y 不可用。
- 只有井口位置且未明确直井：不默认按直井处理。
- 同时提供绝对坐标和偏移量：执行一致性检查并报告误差。

### 10.6 时深关系输入

Checkshot、VSP 或已有时深表推荐使用：

```csv
TVDSS,TWT_MS
1000,650
1100,720
1200,795
```

配置示例：

```yaml
inputs:
  time_depth:
    path: "C:/data/my_sample_001/time_depth.csv"
    format: checkshot
    optional: true
```

时深表必须明确：

- 深度轴为 MD、TVD 或 TVDSS。
- 时间为单程时间还是双程时间。
- 时间单位是否为毫秒。

没有时深表时使用：

```yaml
inputs:
  time_depth:
    path: null
    optional: true
```

若 AC/DT 单位明确且存在可用垂向深度轴，程序可生成近似时深关系。没有井级 t0、替换速度或控制点时，结果只相对声波起测点并标记为低可信度。

### 10.7 缺失输入的写法

允许只处理地震或只处理测井。缺失模态统一写为：

```yaml
inputs:
  seismic:
    path: null
    optional: true
```

不得填写不存在的占位路径，不得用全零文件代替缺失模态。

## 11. 创建实际样本配置

### 11.1 复制模板

```powershell
Copy-Item `
  ".\examples\sample_config.yaml" `
  ".\configs\my_sample_001.yaml"

notepad ".\configs\my_sample_001.yaml"
```

### 11.2 必改字段

每个样本至少检查以下内容：

- `sample_id`：样本唯一名称。
- `task.target_classes`：简短目标类别。
- `inputs.*.path`：原始文件路径。
- `inputs.seismic.domain`：时间域或深度域。
- `inputs.seismic.crs` 和 `coordinate_system.*_crs`：明确 CRS 或 `null`。
- `depth_reference`：测井深度轴、单位、参考面和正方向。
- `processing.well_logs.curve_units`：CSV 曲线单位。
- `processing.time_depth`：时深来源、t0、替换速度和控制点。
- `output.directory`：本次运行的独立输出目录。

Windows YAML 路径推荐使用正斜杠：

```yaml
path: "C:/TT/竞赛/揭榜挂帅/数据集-预处理/example/well.las"
```

相对输入路径优先按命令运行目录解析，找不到时再按配置文件目录解析；输出目录按命令运行目录解析。

### 11.3 最小配置模板

```yaml
schema_version: "1.0"
sample_id: "my_sample_001"

task:
  type: "geological_target_detection"
  target_classes: [fault, channel]

inputs:
  seismic:
    path: "C:/data/my_sample_001/seismic.sgy"
    format: auto
    domain: time
    crs: null
    optional: true
  well_log:
    path: "C:/data/my_sample_001/well_log.las"
    format: auto
    well_id: "WELL_001"
    optional: true
  well_location:
    path: "C:/data/my_sample_001/well_location.csv"
    optional: true
  trajectory:
    path: "C:/data/my_sample_001/trajectory.csv"
    optional: true
  time_depth:
    path: null
    optional: true

coordinate_system:
  project_crs: null
  seismic_crs: null
  well_crs: null
  allow_unknown_crs: true
  require_explicit_crs_for_precise_alignment: true

depth_reference:
  well_log_axis: MD
  unit: m
  reference_surface: KB
  positive_direction: down
  vertical_datum: MSL
  tvdss_sign_convention: positive_below_sea_level

processing:
  seismic:
    views: [inline, crossline, slice, local_patch]
    percentile_clip: {lower: 1.0, upper: 99.0}
    normalization: symmetric
  well_logs:
    short_gap_interpolation:
      enabled: true
      max_gap_samples: 3
      method: linear
  time_depth:
    sonic_integration:
      enabled: true
      preferred_depth_axis: TVDSS
      require_trajectory_for_deviated_well: true
    t0: {policy: per_well, value_ms: null, source: null}
    replacement_velocity: {policy: per_well, value_m_s: null, source: null}
    calibration:
      required_for_joint_analysis: true
      method: null
      control_points_path: null

output:
  directory: "runs/my_sample_001"
  overwrite: true
```

未知的 CRS、t0、替换速度、参考面或标定结果必须保持 `null`，不得为通过流程而填写估计值。

### 11.4 检查并生成

```powershell
geo-adapter inspect --config ".\configs\my_sample_001.yaml"
geo-adapter prepare --config ".\configs\my_sample_001.yaml"
geo-adapter validate --run-dir ".\runs\my_sample_001"
geo-adapter show-manifest --run-dir ".\runs\my_sample_001"
```

`inspect` 阶段应重点确认：

- 九类曲线是否映射到正确原始 mnemonic。
- CSV 单位是否完整。
- 电阻率候选及测量体系是否合理。
- 地震域和 CRS 是否明确。
- 井名是否一致。
- 轨迹和时深关系是否满足联合分析要求。
- 警告是否属于可接受限制。

## 12. Python API

```python
from geo_adapter import inspect_geo_sample, prepare_geo_sample, validate_run

inspection = inspect_geo_sample("configs/my_sample_001.yaml")
result = prepare_geo_sample("configs/my_sample_001.yaml")

if result.success:
    validation = validate_run(result.output_directory)
    print(result.manifest_path, result.run_mode, result.fusion_permission)
else:
    print(result.errors)
```

`PrepareResult` 包含成功状态、输出目录、manifest/request 路径、警告、错误、运行模式、水平/垂向等级和融合权限。

## 13. 标准输出

示例输出位于 `runs/demo_sample_001/`：

```text
input_config.yaml
manifest.json
request.json
assets/seismic/*_model.png, *_qc.png
assets/well_logs/well_log_panel.png, gr.png, ...
arrays/seismic_*.npy
arrays/well_values.npy
arrays/well_valid_mask.npy
arrays/well_interpolated_mask.npy
arrays/curve_available.npy
tables/well_logs_raw.csv
tables/well_logs_clean.csv
tables/well_numeric_summary.json
tables/curve_mapping.csv
tables/well_location_normalized.json
tables/trajectory_normalized.csv
tables/time_depth.csv
prompts/system_prompt.txt
prompts/user_prompt.txt
qc/quality_report.json
qc/processing_log.json
qc/warnings.txt
schemas/manifest.schema.json
schemas/expected_model_output.schema.json
```

仅为实际可用模态生成资产。`manifest.json` 保存源文件 SHA-256、原始路径、配置哈希、曲线来源、转换、候选选择理由和配准限制。`request.json` 只引用实际存在的文件，每张地震图都有独立 `physical_view`。

## 14. 接入 Qwen3-VL

### 14.1 接口边界

本项目负责生成标准模型输入包，不负责下载、部署或调用 Qwen3-VL。`request.json` 是推理框架无关的请求清单，不是 Transformers、vLLM 或云端 API 的原生请求格式。

Qwen3-VL 不直接接收 SEG-Y、LAS、CSV 或 NPY。模型调用端应从运行目录加载：

- `request.json` 中每个地震图的 `analysis_path`：带坐标轴、色标、原生网格尺寸的 VLM 分析图；下游模型仍使用 `path` 指向的无标注图。
- `assets/well_logs/well_log_panel.png`：测井综合图。
- `tables/well_numeric_summary.json`：由清洗后的结构化测井表生成的曲线统计和代表性采样点，是数值结论的权威来源。
- `prompts/system_prompt.txt`：系统约束。
- `prompts/user_prompt.txt`：当前样本任务说明。
- `manifest.json`：曲线、单位、深度、CRS、时深关系和限制。
- `schemas/expected_model_output.schema.json`：模型输出格式。

`arrays/*.npy` 和 `tables/*.csv` 用于数值复核与下游程序，当前 `request.json` 不把这些二进制数组直接传给视觉语言模型。地震 PNG 只支持构造、连续性和位置判断，不能替代原始振幅体；测井 PNG 只支持趋势判断，精确数值必须来自结构化摘要。

### 14.2 `request.json` 转换规则

调用端按以下规则转换：

| request 内容 | Qwen 消息内容 |
|---|---|
| `type: image` | 解析为绝对路径并转换成 `file:///...` 图片输入 |
| `type: text` | 读取 `text_path` 的 UTF-8 文本 |
| `type: json` | 读取 JSON，序列化为带说明的文本块 |
| `expected_output_schema` | 读取 Schema，追加到用户文本并用于本地结果校验 |

不同地震视图不得合成 RGB。为避免多图注意力稀释，推荐每次只分析一个 `physical_view`，再在程序侧合并各视图证据。图片名称、`physical_view`、`native_shape`、`axis_labels` 和 `source_indices` 应同时写入文字上下文。

### 14.3 Qwen Transformers 本地推理参考

Qwen3-VL 官方 Transformers 接口使用 `AutoModelForImageTextToText` 和 `AutoProcessor`，并支持多张本地图片。模型运行环境与本项目依赖分开安装：

官方参考：[QwenLM/Qwen3-VL Quickstart](https://github.com/QwenLM/Qwen3-VL#quickstart)

```powershell
python -m pip install "transformers>=4.57.0" accelerate
```

PyTorch 应根据本机 CUDA、显卡驱动和操作系统单独安装。模型权重会占用较大磁盘和显存；示例使用较小的 `Qwen/Qwen3-VL-2B-Instruct`，实际型号由部署环境决定。

以下代码演示如何把运行包转换为 Qwen 消息。该代码是接入参考，当前仓库尚未提供 `geo-adapter infer` 命令：

```python
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from transformers import AutoModelForImageTextToText, AutoProcessor


RUN_DIR = Path("runs/my_sample_001").resolve()
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def read_text(relative_path: str) -> str:
    return (RUN_DIR / relative_path).read_text(encoding="utf-8")


def build_messages() -> tuple[list[dict], dict]:
    request = json.loads(read_text("request.json"))
    messages: list[dict] = []

    for message in request["messages"]:
        converted_content: list[dict] = []
        for item in message["content"]:
            item_type = item["type"]

            if item_type == "image":
                image_path = (RUN_DIR / item["path"]).resolve()
                converted_content.append(
                    {
                        "type": "image",
                        "image": image_path.as_uri(),
                    }
                )
                converted_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"上一张图片名称为 {item['name']}，"
                            f"物理观察方向为 {item['physical_view']}。"
                        ),
                    }
                )

            elif item_type == "text":
                converted_content.append(
                    {
                        "type": "text",
                        "text": read_text(item["text_path"]),
                    }
                )

            elif item_type == "json":
                payload = json.loads(read_text(item["path"]))
                converted_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"结构化上下文 {item.get('name', 'json')}：\n"
                            + json.dumps(payload, ensure_ascii=False, indent=2)
                        ),
                    }
                )

        messages.append(
            {
                "role": message["role"],
                "content": converted_content,
            }
        )

    output_schema = json.loads(read_text(request["expected_output_schema"]))
    messages[-1]["content"].append(
        {
            "type": "text",
            "text": (
                "输出必须严格满足以下 JSON Schema：\n"
                + json.dumps(output_schema, ensure_ascii=False, indent=2)
            ),
        }
    )
    return messages, output_schema


def parse_json_output(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(cleaned)


messages, output_schema = build_messages()

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=2048)
trimmed_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    trimmed_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

model_output = parse_json_output(output_text)
jsonschema.validate(model_output, output_schema)

(RUN_DIR / "model_output.json").write_text(
    json.dumps(model_output, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(RUN_DIR / "model_output.json")
```

### 14.4 vLLM 或 OpenAI 兼容服务

使用 vLLM、本地 OpenAI 兼容服务或云端 API 时，仍按 14.2 节加载请求包，但消息图片格式通常需要转换为 `image_url`：

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,..."
  }
}
```

远程服务不能直接访问本机 `C:\...` 路径。调用端必须将图片转换为 Base64 data URL、上传到服务可访问的受控地址，或使用服务商提供的文件接口。涉及真实井、坐标和地震资料时，应先确认数据授权和保密要求，不得默认上传到第三方服务。

### 14.5 模型输出验收

模型输出至少执行以下检查：

1. 能解析为单个 JSON 对象。
2. 通过 `expected_model_output.schema.json` 校验。
3. `sample_id` 与运行包一致。
4. bbox 为 0～1 的归一化 `xyxy`。
5. 深度区间包含 MD/TVD/TVDSS 类型。
6. 融合权限不足时，`cross_modal_analysis.allowed` 为 `false`。
7. `class_prompts` 使用简短检测类别词。

模型自然语言推理不等于可靠地质标签。所有候选异常、井震对应和下游检测区域仍需专业人员及独立证据复核。

## 15. 交给 YOLO-World

YOLO-World 通常读取地震 PNG 和简短类别 Prompt，例如 `fault`、`channel`，而不是整段地质分析。下游可读取模型 JSON 的 `downstream_plan`：

- `input_images`：模型图名称；
- `class_prompts`：简短检测词；
- `regions_of_interest`：归一化 `xyxy`；
- `confidence_threshold`：检测阈值。

通用 YOLO-World 对专业地震目标可能没有足够的领域识别能力，通常需要专业标注数据、微调或专用检测器支持。候选异常不能直接当作油气藏标签。

## 16. 测试

```powershell
pytest -q
```

测试覆盖九槽位、SP 整条缺失、多个深电阻率候选、感应/侧向/微聚焦、非正电阻率、NPHI 小数/百分数、AC 两种单位、短/长缺口、明确/未知 CRS、经纬度/投影坐标、三种轨迹输入、Checkshot、未标定/已标定 AC、无时深关系、六种运行模式、Prompt、manifest、request、Mask 和完整 CLI 端到端输出。

## 17. 已知限制与保守假设

- NumPy 三维体采用 `[inline, crossline, sample]` 约定；如果源轴顺序不同，调用方应先转换。
- NumPy 示例只具有索引坐标；没有真实 inline/crossline 坐标数组时，不输出伪造的物理坐标映射或井位叠加图。
- 第一版 SEG-Y 读取依赖可建立的 inline/crossline 几何，且只实现按需剖面提取；不可靠几何会明确报错。
- 地方工程坐标系、Beijing 1954、Xi'an 1980、CGCS 2000 等必须给出可执行的 EPSG/PROJ 参数；名称本身不足以转换。
- 控制点标定是透明的仿射校正，只在控制点覆盖范围内解释；未实现默认自动相关井震匹配。
- 基础合成地震记录不自动宣称完成井震标定，也不替代 Checkshot/VSP。
- 未明确井为直井时，不会因缺轨迹而假设直井。
- `t0` 和替换速度始终按井/样本配置，不存在全局正式默认值。
