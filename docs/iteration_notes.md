> **归档说明（2026-07-21）**：本文档记录的是 Qwen3.5-9B 时代的 prompt 调参笔记，
> 项目已切换到 Qwen3-VL-8B + 11 下游模型闭环架构。保留作历史参考，不代表当前实现。
> 最新架构见 README.md 与 docs/PIPELINE_TECHNICAL_DOC.md。

# 迭代记录 — VLM Prompt 测试与优化

## 测试环境
- **模型**: Qwen3.5-9B (Qwen3_5ForConditionalGeneration, 9.4B params, VLM)
- **加载方式**: transformers 直接推理 (非 vLLM, vLLM v0.21.0 与 CUDA 12.9 不兼容)
- **GPU**: 1x A100 80GB
- **推理参数**: `do_sample=True, temperature=0.3, repetition_penalty=1.1, top_p=0.9`

## 关键发现

### 1. 生成参数至关重要
| 问题 | 原因 | 解决方案 |
|------|------|----------|
| JSON被截断 | max_tokens太少 | 图像Agent: 6144, 文本Agent: 10240 |
| 重复输出 | do_sample=False (贪心解码) | 用 do_sample=True + temperature=0.3 |
| 陷入循环 | 无惩罚 | repetition_penalty=1.1 |
| `<think>`太长 | Qwen3.5模型固有行为 | post-process: 提取`</think>`之后的内容 |

### 2. Prompt 风格
- ✅ 短 prompt + 明确的输出目标 → 模型响应更快且JSON更完整
- ✅ 在 system prompt 中嵌入领域知识表格（如测井响应特征表） → 模型能正确引用
- ✅ 在 user_text 中直接给出JSON keys 示例 → 模型输出格式准确
- ❌ 冗长的角色扮演描述 → 浪费 token 在思考过程中

### 3. 各Agent的表现

**SeismicInterpAgent**:
- 能正确识别模拟数据中的断层、层位和异常体
- 对"无断层"的情况也能正确返回空数组
- 建议: 对真实SEG-Y数据需要真实的色标和坐标

**LogAnalysisAgent**:
- 准确识别了砂泥岩互层序列
- 正确判断含气层段(高RT+低DEN+低CNL组合)
- 建议: 6条曲线图需要足够大的DPI (≥120) 以保证VLM可读

**WellSeismicFusionAgent**:
- 能建立合理的时深关系(即使用模拟数据)
- 输出的相关系数(r=0.82)和标定质量在合理范围
- 建议: 井震对比图需要并排布局，左侧地震右侧测井

**ProspectEvaluationAgent**:
- 正确评估两个圈闭的风险等级
- T1背斜 → drill_ready (Pg=70%)，T2断块 → inventory (Pg=45%)
- 风险评估合理：T2圈闭风险更高(闭合仅30ms)
- 建议: 这是纯文本Agent，prompt中嵌入风险矩阵可提升评估一致性

## 后续优化方向

1. **真实数据测试**: 用 SEG-Y + LAS 真实数据替换模拟数据
2. **多轮对话**: Agent之间通过JSON传递结果(而非文本摘要)
3. **图像预处理优化**: 地震色标、测井图布局的标准化
4. **减少延迟**: 考虑用 vLLM 替代 transformers (需更新CUDA驱动)
5. **Prompt压缩**: 将领域知识表格移到RAG，减少system prompt长度
