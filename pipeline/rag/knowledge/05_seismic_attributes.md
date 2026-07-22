# 地震属性知识

## 属性分类与地质含义

### 瞬时属性 (Hilbert变换)
- **envelope (瞬时振幅)**: 反射强度, 亮点/暗点检测, 储层厚度变化
- **phase (瞬时相位)**: 层位连续性, 不整合面, 断层
- **frequency (瞬时频率)**: 薄层检测(高频→薄层), 含气衰减(低频阴影)

### 纹理属性 (GLCM)
- **energy (能量)**: 高值→连续平行反射, 低值→杂乱
- **contrast (对比度)**: 高值→河道/断层边界
- **homogeneity (均质性)**: 高值→均匀沉积, 低值→复杂构型

### 复合属性
- **sweetness (甜点)**: envelope/√frequency, 高值→储层发育段
- **RMS amplitude**: 滑动窗能量, 砂体厚度指示

## 下游模型选择: attribute_extractor
- attributes: 选择需要的属性列表
- 推荐组合:
  - 储层预测: ["envelope","sweetness","rms_amplitude"]
  - 断层辅助: ["frequency","phase"]
  - 沉积相: ["envelope","frequency","sweetness"]

## 裂缝检测
- 首选: seismic_domain_model (attribute=coherence|variance)
- 裂缝特征: 高密度不连续反射带、相干性异常、局部振幅衰减
- conf_threshold建议: 0.2-0.3 (裂缝信号弱于断层)
