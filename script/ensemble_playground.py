import argparse
import json
import os
import numpy as np


def load_scores(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing score file: {path}")
    data = np.load(path, allow_pickle=True)
    logits = data["logits"]
    labels = data["labels"]
    names = data["names"]
    return logits, labels, names


def weighted_fuse(logits_list, weights):
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    fused = np.zeros_like(logits_list[0], dtype=np.float32)
    for logit, weight in zip(logits_list, w):
        fused += logit * weight
    return fused


def compute_metrics(logits, labels):
    preds = logits.argmax(axis=1)
    acc = (preds == labels).mean().item()
    num_class = logits.shape[1]
    cm = np.zeros((num_class, num_class), dtype=int)
    for y, p in zip(labels, preds):
        cm[y, p] += 1
    precision, recall, f1 = [], [], []
    for c in range(num_class):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1_c = 2 * prec * rec / (prec + rec + 1e-6)
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_c)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }, preds


def main():
    parser = argparse.ArgumentParser(description="Late fusion ensemble for playground MP-GCN.")
    parser.add_argument("--fold", required=True, help="Fold id, e.g., 06")
    parser.add_argument("--streams", nargs="+", required=True, help="Stream types, e.g., J B JM BM")
    parser.add_argument("--weights", nargs="+", type=float, help="Optional weights matching streams")
    parser.add_argument("--workdir", default="./workdir", help="Base workdir containing fold_*_{stream}")
    args = parser.parse_args()

    streams = args.streams
    weights = args.weights if args.weights else [1.0] * len(streams)
    if len(weights) != len(streams):
        raise ValueError("weights length must match streams length")

    logits_list, labels_ref, names_ref = [], None, None
    for stream in streams:
        fold_dir = f"fold_{args.fold}_{stream}"
        score_path = os.path.join(args.workdir, fold_dir, "score_eval.npz")
        logits, labels, names = load_scores(score_path)
        if labels_ref is None:
            labels_ref, names_ref = labels, names
        else:
            if not np.array_equal(labels_ref, labels):
                raise ValueError(f"Label mismatch for stream {stream}")
            if not np.array_equal(names_ref, names):
                raise ValueError(f"Name/order mismatch for stream {stream}")
        logits_list.append(logits)

    fused_logits = weighted_fuse(logits_list, weights)
    metrics, preds = compute_metrics(fused_logits, labels_ref)

    out_dir = os.path.join(args.workdir, f"fold_{args.fold}_ensemble")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"fused_score_eval_fold{args.fold}.json"), "w") as f:
        json.dump(
            {
                "streams": streams,
                "weights": weights,
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    np.savez(
        os.path.join(out_dir, f"fused_logits_fold{args.fold}.npz"),
        logits=fused_logits,
        labels=labels_ref,
        names=names_ref,
        preds=preds,
    )
    print(f"Ensemble complete. Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
