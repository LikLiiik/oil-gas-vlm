# 沉积相分析知识

## 地震反射构型与沉积相

| 反射构型 | 沉积环境解释 | 典型特征 |
|----------|-------------|---------|
| 平行/亚平行 | 陆棚稳定沉积 | 振幅均匀,连续性好 |
| S形前积 | 三角洲前缘 | 顶超+底超,倾斜反射 |
| 斜交前积 | 高能三角洲 | 陡倾角,顶超明显 |
| 杂乱反射 | 生物礁/碎屑流/滑塌 | 不连续,振幅多变 |
| 丘状反射 | 浊积扇/深海扇 | 透镜状,双向尖灭 |
| 透镜状+侧向尖灭 | 河道充填 | 凹形底界,内部杂乱或平行 |

## 下游模型选择

### 首选: facies_classifier
- 算法: 多属性PCA + GMM聚类
- 参数: n_clusters=3-6 (根据区域复杂度), method=gmm
- 推荐先用 attribute_extractor 提取属性再聚类
- attribute_list: ["envelope","frequency","sweetness"] (基础组合)
  或 ["envelope","phase","gradient_magnitude","local_variance"] (详细组合)

### 辅助: seismic_foundation
- SFM seismic预训练ViT提取特征
- task: feature_extraction | facies_classification

### 辅助: attribute_extractor
- 先提取 envelope+sweetness+frequency
- 再用 facies_classifier 聚类

## 沉积相分类参数建议

- 简单沉积环境(2-3种相): n_clusters=3, method=kmeans
- 复杂沉积环境(4-6种相): n_clusters=5, method=gmm
- 河道/浊积水道: 考虑用 cig_channel 检测
