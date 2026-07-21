"""领域知识检索器 — TF-IDF 检索地球物理知识文档。

在 VLM 规划前，根据任务关键词检索相关知识，
注入到 prompt 中辅助 VLM 决策。
"""
from __future__ import annotations

import re
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


class DomainRetriever:
    """TF-IDF 知识检索器。加载 knowledge/ 下所有 .md 文件，
    根据 query 检索 top_k 最相关文档。
    """

    def __init__(self):
        self.docs: dict[str, str] = {}      # doc_id → content
        self._idf: dict[str, float] = {}    # term → idf
        self._tf: dict[str, dict[str, float]] = {}  # doc_id → {term: tf}
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        for f in sorted(KNOWLEDGE_DIR.glob("*.md")):
            doc_id = f.stem
            text = f.read_text(encoding="utf-8")
            self.docs[doc_id] = text
        self._build_index()
        self._loaded = True

    def _tokenize(self, text: str) -> list[str]:
        """中文+英文混合分词。"""
        # 保留中文字符、英文单词、数字
        tokens = re.findall(r'[一-鿿]+|[a-zA-Z_]+|\d+\.?\d*',
                            text.lower())
        # 过滤停用词
        stop = {'the', 'a', 'an', 'is', 'are', 'of', 'to', 'in', 'for',
                'and', 'or', 'on', 'at', 'with', 'by', 'from', '的', '了',
                '在', '是', '和', '与', '或', '等', '及', '之'}
        return [t for t in tokens if t not in stop and len(t) > 1]

    def _build_index(self):
        """构建 TF-IDF 索引。"""
        N = len(self.docs)
        doc_terms = {}
        df = {}  # document frequency

        for doc_id, text in self.docs.items():
            terms = self._tokenize(text)
            doc_terms[doc_id] = terms
            for t in set(terms):
                df[t] = df.get(t, 0) + 1

        # IDF
        import math
        self._idf = {t: math.log((N + 1) / (df[t] + 1)) + 1
                     for t in df}

        # TF (normalized)
        for doc_id, terms in doc_terms.items():
            total = len(terms) or 1
            tf = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
            self._tf[doc_id] = {t: v / total for t, v in tf.items()}

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """检索与 query 最相关的 top_k 篇文档。

        返回 [{doc_id, content, score}]
        """
        self._load()
        q_terms = self._tokenize(query)
        if not q_terms:
            return []

        # TF for query
        q_tf = {}
        for t in q_terms:
            q_tf[t] = q_tf.get(t, 0) + 1
        total = len(q_terms) or 1
        q_tf = {t: v / total for t, v in q_tf.items()}

        scores = {}
        for doc_id in self.docs:
            score = 0.0
            doc_tf = self._tf.get(doc_id, {})
            for t, qv in q_tf.items():
                dv = doc_tf.get(t, 0)
                idf = self._idf.get(t, 0)
                score += qv * dv * idf
            if score > 0:
                scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [{"doc_id": did, "content": self.docs[did], "score": round(s, 4)}
                for did, s in ranked]


# 全局单例
_retriever: DomainRetriever | None = None


# 关键词→文档映射 (显式规则兜底)
_KEYWORD_MAP = {
    "fault":     ["01_fault_detection"],
    "断层":       ["01_fault_detection"],
    "horizon":   ["02_horizon_tracking"],
    "层位":       ["02_horizon_tracking"],
    "facies":    ["03_facies_analysis"],
    "沉积相":     ["03_facies_analysis"],
    "channel":   ["03_facies_analysis"],
    "河道":       ["03_facies_analysis"],
    "reservoir": ["03_facies_analysis", "05_seismic_attributes"],
    "储层":       ["03_facies_analysis", "05_seismic_attributes"],
    "fracture":  ["05_seismic_attributes", "01_fault_detection"],
    "裂缝":       ["05_seismic_attributes", "01_fault_detection"],
    "log":       ["04_well_log_analysis"],
    "测井":       ["04_well_log_analysis"],
    "well":      ["04_well_log_analysis"],
    "lithology": ["04_well_log_analysis"],
    "岩性":       ["04_well_log_analysis"],
    "fluid":     ["04_well_log_analysis"],
    "流体":       ["04_well_log_analysis"],
    "gas":       ["04_well_log_analysis"],
    "oil":       ["04_well_log_analysis"],
    "attribute": ["05_seismic_attributes"],
    "属性":       ["05_seismic_attributes"],
    "envelope":  ["05_seismic_attributes"],
    "sweetness": ["05_seismic_attributes"],
    "seismic":   ["05_seismic_attributes"],
}


def retrieve_for_task(target_classes: list[str],
                      image_views: list[str] | None = None,
                      top_k: int = 3) -> str:
    """根据任务关键词检索知识，返回可注入 prompt 的文本。

    采用关键词精确匹配 + TF-IDF 混合检索。
    """
    global _retriever
    if _retriever is None:
        _retriever = DomainRetriever()

    # 1) 关键词精确匹配
    doc_ids = []
    seen = set()
    query_parts = list(target_classes)
    if image_views:
        query_parts.extend(image_views)
    for cls in query_parts:
        for kw, ids in _KEYWORD_MAP.items():
            if kw.lower() in cls.lower():
                for did in ids:
                    if did not in seen:
                        doc_ids.append(did)
                        seen.add(did)

    # 2) 补充 TF-IDF 检索
    query = " ".join(query_parts)
    tfidf_results = _retriever.retrieve(query, top_k=top_k + 2)
    for r in tfidf_results:
        if r["doc_id"] not in seen and len(doc_ids) < top_k:
            doc_ids.append(r["doc_id"])
            seen.add(r["doc_id"])

    if not doc_ids:
        return ""

    lines = ["\n=== 领域知识参考 (RAG检索) ===\n"]
    for did in doc_ids[:top_k]:
        content = _retriever.docs.get(did, "")
        if not content:
            continue
        lines.append(f"--- {did} ---")
        if len(content) > 1500:
            content = content[:1500] + "\n...(截断)"
        lines.append(content)
        lines.append("")
    return "\n".join(lines)
