"""Facies Classifier — 多属性沉积相无监督分类。

VLM 描述期望的沉积相特征 → 计算多属性 → PCA 降维 → GMM/KMeans 聚类 →
返回簇中心供 VLM 解释为地质相。

每个簇返回其属性特征，VLM 可以根据地质知识解释：
  - 高能/连续/强振幅 → 可能的平行席状砂相
  - 低能/不连续/弱振幅 → 可能的深水泥岩相
  - 高瞬时频率/杂乱 → 可能的河道充填相
"""
from __future__ import annotations

import numpy as np

from ._shared import image_to_array as _get_array


def _extract_attribute_stack(arr: np.ndarray,
                             attr_list: list[str] | None = None,
                             ) -> np.ndarray:
    """从 2D 地震数组提取多属性体 (n_attrs, H, W)。

    如果 scipy 不可用，则退化到用局部窗口的统计量作为属性。
    """
    ns, nt = arr.shape
    try:
        from scipy.signal import hilbert
        from scipy.ndimage import sobel, uniform_filter, gaussian_filter
    except ImportError:
        # 退化：只用局部统计量
        from scipy.ndimage import uniform_filter
        feats = []
        for w in (5, 11, 21):
            feats.append(uniform_filter(arr, size=w))
            feats.append(uniform_filter(arr * arr, size=w)
                         - uniform_filter(arr, size=w) ** 2)
        return np.stack(feats, axis=0).astype(np.float32)

    if attr_list is None:
        attr_list = ["envelope", "gradient", "local_variance"]

    feats: list[np.ndarray] = []

    for an in attr_list:
        try:
            if an == "envelope":
                analytic = hilbert(arr, axis=0)
                feats.append(np.abs(analytic).astype(np.float32))
            elif an == "phase":
                analytic = hilbert(arr, axis=0)
                feats.append(np.angle(analytic).astype(np.float32))
            elif an == "frequency":
                analytic = hilbert(arr, axis=0)
                ph = np.angle(analytic)
                dph = np.diff(np.unwrap(ph, axis=0), axis=0)
                dph = np.vstack([dph, dph[-1:, :]])
                feats.append(np.clip(np.abs(dph), 0, 50).astype(np.float32))
            elif an == "gradient_magnitude":
                gx = sobel(arr, axis=1, mode="reflect")
                gy = sobel(arr, axis=0, mode="reflect")
                feats.append(np.sqrt(gx ** 2 + gy ** 2).astype(np.float32))
            elif an == "local_variance":
                mu = uniform_filter(arr, size=9, mode="reflect")
                mu2 = uniform_filter(arr * arr, size=9, mode="reflect")
                feats.append(np.clip(mu2 - mu * mu, 0, None).astype(np.float32))
            elif an == "local_entropy":
                # 局部直方图熵（简化）
                smoothed = gaussian_filter(arr, sigma=1.5, mode="reflect")
                ent = np.zeros_like(arr, dtype=np.float32)
                win = 5
                for i in range(win, ns - win, 1):
                    for j in range(win, nt - win, 1):
                        patch = smoothed[i - win:i + win, j - win:j + win]
                        hist, _ = np.histogram(patch, bins=8,
                                               range=(patch.min(), patch.max()))
                        prob = hist / (hist.sum() + 1e-8)
                        ent[i, j] = -np.sum(prob * np.log(prob + 1e-8))
                feats.append(ent.astype(np.float32))
            elif an == "dip":
                # 瞬时倾角（相位梯度）
                analytic = hilbert(arr, axis=0)
                ph = np.angle(analytic)
                gx, gy = np.gradient(ph)
                dip = np.arctan2(gy, gx)
                feats.append(dip.astype(np.float32))
        except Exception:
            continue

    # 如果没有成功计算任何属性，用原数组兜底
    if not feats:
        feats = [arr]
    return np.stack(feats, axis=0).astype(np.float32)


# ── 下游模型 ────────────────────────────────────────────────────────────────

class FaciesClassifier:
    name = "facies_classifier"
    description = (
        "多属性沉积相分类。计算多个地震属性→PCA降维→GMM/KMeans聚类，"
        "返回各类的空间分布和属性特征，供VLM进行地质相解释。"
        "适合沉积相图编制和储层预测"
    )
    required_fields = [
        "n_clusters (聚类簇数，建议3-8)",
        "attribute_list? (可选属性列表: envelope, phase, frequency, "
        "gradient_magnitude, local_variance, local_entropy)",
        "regions_of_interest? (可选ROI列表)",
        "method? (聚类方法: kmeans|gmm，默认gmm)",
    ]
    output_shape = (
        "list[{id, cluster_id, area_pixels, centroid_xy, "
        "cluster_center: {attr_values}, dominant_feature}]"
    )

    def detect(self, instruction: dict, image=None,
               context: dict | None = None) -> list[dict]:
        arr = _get_array(image, context)
        if arr is None:
            return []

        n_clusters = int(instruction.get("n_clusters", 4))
        attr_list = instruction.get("attribute_list")
        method = instruction.get("method", "gmm")
        rois_raw = instruction.get("regions_of_interest") or []
        ns, nt = arr.shape

        # ── 提取多属性 ──
        attr_stack = _extract_attribute_stack(arr, attr_list)  # (n_attr, H, W)
        n_attr, ny, nx = attr_stack.shape

        # ── 确定分析区域 ──
        if rois_raw:
            mask = np.zeros((ny, nx), dtype=bool)
            for roi in rois_raw:
                bn = roi.get("bbox_norm") or roi.get("bbox_xyxy_norm")
                if not bn or len(bn) != 4:
                    continue
                x1 = int(np.clip(bn[0] * nx, 0, nx - 1))
                y1 = int(np.clip(bn[1] * ny, 0, ny - 1))
                x2 = int(np.clip(bn[2] * nx, 0, nx - 1))
                y2 = int(np.clip(bn[3] * ny, 0, ny - 1))
                mask[y1:y2 + 1, x1:x2 + 1] = True
        else:
            mask = np.ones((ny, nx), dtype=bool)

        # ── 构建特征矩阵 ──
        ys, xs = np.where(mask)
        if len(ys) < n_clusters * 10:
            return []

        X = np.column_stack([attr_stack[:, y, x] for y, x in zip(ys, xs)]).T
        # (n_pixels, n_attrs)

        # Z-score normalize
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X_norm = (X - mean) / std

        # ── PCA 降维（保留 95% 方差） ──
        if n_attr > 2:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(n_attr, 3))
                X_red = pca.fit_transform(X_norm)
            except ImportError:
                X_red = X_norm[:, :min(n_attr, 3)]
        else:
            X_red = X_norm

        # ── 聚类 ──
        labels = np.zeros(len(ys), dtype=np.int32)
        try:
            if method == "kmeans":
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=n_clusters, random_state=0, n_init=3)
                labels = km.fit_predict(X_red)
                centers = km.cluster_centers_
            else:
                from sklearn.mixture import GaussianMixture
                gmm = GaussianMixture(n_components=n_clusters,
                                      random_state=0, n_init=3)
                labels = gmm.fit_predict(X_red)
                centers = gmm.means_
        except ImportError:
            # 退化到简单分位数聚类
            from scipy.cluster.vq import kmeans2
            centers, labels = kmeans2(X_red.astype(np.float64), n_clusters,
                                      iter=10, minit="points")

        # ── 构建结果 ──
        results: list[dict] = []
        for cid in range(n_clusters):
            c_mask = labels == cid
            if c_mask.sum() < 10:
                continue
            c_ys = ys[c_mask]
            c_xs = xs[c_mask]
            # 该簇在每个属性上的均值
            attr_means = X[c_mask].mean(axis=0)
            attr_dict = {
                f"attr_{i}": round(float(attr_means[i]), 4)
                for i in range(min(n_attr, len(attr_means)))
            }
            # 找最具区分性的属性（Z-score 最大的一项）
            z_vals = [
                abs(float((attr_means[i] - mean[0, i]) / (std[0, i] + 1e-8)))
                for i in range(n_attr)
            ]
            dominant_idx = int(np.argmax(z_vals))
            dom_name = (attr_list[dominant_idx]
                        if attr_list and dominant_idx < len(attr_list)
                        else f"attr_{dominant_idx}")

            results.append({
                "id": f"facies_cluster_{cid}",
                "cluster_id": int(cid),
                "area_pixels": int(c_mask.sum()),
                "centroid_xy": [float(c_xs.mean()), float(c_ys.mean())],
                "cluster_center": attr_dict,
                "dominant_feature": dom_name,
                "n_attributes_used": n_attr,
                "model": self.name,
            })
        return results
