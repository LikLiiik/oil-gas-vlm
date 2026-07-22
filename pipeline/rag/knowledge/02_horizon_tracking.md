# 层位追踪知识

## 层位识别标志

1. **强连续反射**: 横向上波形/振幅稳定的同相轴，代表地层界面
2. **上超/下超**: 反射终止于不整合面之上/之下
3. **顶超**: 倾斜反射向上倾方向尖灭于层序顶面

## 下游模型选择

### 首选: horizon_tracker
- 算法: 互相关追踪 (np.correlate)
- 输入: 2D切片 + seed_points[{trace_idx, sample_idx}]
- 参数: tracking_mode=correlation (推荐, 最鲁棒)
- 备选模式: peak(波峰), trough(波谷), zero_crossing(零交叉)
- 推荐 search_window_samples=15

## 追踪失败处理

- 断层处自动终止(相关性<0.4)
- 信噪比低区域可能跳相位 → 降低search_window_samples
- 倾斜层位→用correlation模式, 比peak/trough更鲁棒
