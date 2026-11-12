"""Tính toán các chỉ số đánh giá cho bài toán phân loại đa nhãn."""

from __future__ import annotations

import numpy as np


def compute_multilabel_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Tính Macro F1, Micro F1, Precision, Recall và Subset Accuracy cho nhãn đa lớp."""
    preds = (probs >= threshold).astype(int)
    targets = (targets >= 0.5).astype(int)

    # Micro F1
    tp_total = np.sum((preds == 1) & (targets == 1))
    fp_total = np.sum((preds == 1) & (targets == 0))
    fn_total = np.sum((preds == 0) & (targets == 1))

    micro_precision = float(tp_total / (tp_total + fp_total + 1e-8))
    micro_recall = float(tp_total / (tp_total + fn_total + 1e-8))
    micro_f1 = float(2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8))

    # Macro F1
    num_classes = targets.shape[1]
    f1_list = []
    ap_list = []

    for c in range(num_classes):
        y_true = targets[:, c]
        y_pred = preds[:, c]
        y_score = probs[:, c]

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = (2 * prec * rec) / (prec + rec + 1e-8) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

        # Xấp xỉ Average Precision nếu có mẫu dương
        if np.sum(y_true) > 0:
            order = np.argsort(-y_score)
            y_eval = y_true[order]
            tp_cumsum = np.cumsum(y_eval == 1)
            precision_curve = tp_cumsum / (np.arange(1, len(y_eval) + 1))
            ap = np.sum(precision_curve * (y_eval == 1)) / np.sum(y_true)
            ap_list.append(ap)

    macro_f1 = float(np.mean(f1_list))
    mean_ap = float(np.mean(ap_list)) if ap_list else 0.0

    # Subset accuracy (exact match)
    subset_acc = float(np.mean(np.all(preds == targets, axis=1)))

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "mAP": round(mean_ap, 4),
        "subset_accuracy": round(subset_acc, 4),
    }
