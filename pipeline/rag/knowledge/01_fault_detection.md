# 断层检测知识

## 地震剖面上断层的识别标志

1. **同相轴错断**: 反射同相轴在断层两侧发生垂直或倾斜位移，是最直接的断层标志
2. **反射终止**: 同相轴在断层面上突然终止，表现为削截(上盘)或上超(下盘)
3. **断面波**: 断层面本身产生的反射，通常为倾斜的弱-中等振幅反射
4. **牵引构造**: 断层两侧同相轴倾角突变，形成拖曳褶皱
5. **同相轴分叉/合并**: 断层破碎带导致反射特征改变

## 下游模型选择指南

### 首选: cig_fault (CIG-Bench FaultPredictor)
- 权重: HRNet在合成+真实地震数据上训练
- 输入: 3D地震体 (T, H, W)
- 参数: threshold(概率阈值, 默认0.5), scale(缩放系���, 默认1.0)
- 适用: 有3D地震体数据时
- 注意: 需要GPU, 首次自动下载权重

### 次选: seismic_domain_model
- 算法: 相干体 + 结构张量 + 梯度断层概率
- 输入: 2D切片数组 或 PIL图像
- 参数: task=fault_detection, attribute=gradient|coherence|structure_tensor|variance
- attribute选择:
  - gradient: 近垂直断层, AGC数据推荐
  - coherence: 任意倾角断层, 通用性最好
  - structure_tensor: 低信噪比数据, 抗噪性强
  - variance: AGC处理后的数据
- 推荐 confidence_threshold=0.3, min_region_area_pixels=80

## 假阳性常见来源

- 河道边缘反射终止(单侧, 非贯穿性)
- 处理噪声条带(横向延伸, 无地质意义)
- 采集脚印(规则间距)
- 振幅渐变(无错断, 可能为岩性变化)

## 断层参数评估

- 断距: 小(<10ms)、中(10-30ms)、大(>30ms)
- 断层类型: normal(正断层, 上盘下降)、reverse(逆断层, 上盘上升)、strike-slip(走滑)
