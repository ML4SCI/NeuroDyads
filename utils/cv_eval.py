import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


PAIR_LABELS = {
    "spk9-lst10": 0,
    "lst9-spk10": 1,
}
METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1")


def normalize_train_fractions(train_fractions):
    fractions = []
    for fraction in train_fractions:
        value = float(fraction)
        if value <= 0.0 or value > 1.0:
            raise ValueError("Training fractions must be in the interval (0, 1].")
        fractions.append(round(value, 6))
    return sorted(set(fractions))


def build_pooled_dataset(run_embeddings_by_pair):
    missing_pairs = [pair for pair in PAIR_LABELS if pair not in run_embeddings_by_pair]
    if missing_pairs:
        raise ValueError(f"Missing embeddings for pairings: {', '.join(missing_pairs)}")

    features = []
    labels = []
    sample_rows = []

    for pair_name, label in PAIR_LABELS.items():
        embedding = np.asarray(run_embeddings_by_pair[pair_name], dtype=np.float32)
        if embedding.ndim != 2:
            raise ValueError(f"Expected 2D embedding array for {pair_name}, got {embedding.ndim}D.")

        features.append(embedding)
        pair_labels = np.full(embedding.shape[0], label, dtype=np.int8)
        labels.append(pair_labels)

        for sample_index in range(embedding.shape[0]):
            sample_rows.append(
                {
                    "pair": pair_name,
                    "class_label": int(label),
                    "sample_index": int(sample_index),
                }
            )

    return np.vstack(features), np.concatenate(labels), sample_rows


def run_repeated_stratified_cv(X, y, folds=5, repeats=1, random_state=0, run_id=None):
    n_splits = _resolve_n_splits(y, folds)
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=max(1, int(repeats)),
        random_state=random_state,
    )

    fold_rows = []
    for split_index, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repeat_index = split_index // n_splits
        fold_index = split_index % n_splits
        predictions = _fit_and_predict(X[train_idx], y[train_idx], X[test_idx])

        fold_rows.append(
            {
                "run": int(run_id) if run_id is not None else "",
                "repeat": int(repeat_index),
                "fold": int(fold_index),
                "n_splits": int(n_splits),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                **_compute_metrics(y[test_idx], predictions),
            }
        )

    return fold_rows


def run_learning_curve(X, y, folds=5, repeats=1, train_fractions=None, random_state=0, run_id=None):
    fractions = normalize_train_fractions(train_fractions or (0.2, 0.4, 0.6, 0.8, 1.0))
    n_splits = _resolve_n_splits(y, folds)
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=max(1, int(repeats)),
        random_state=random_state,
    )

    fold_rows = []
    for split_index, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repeat_index = split_index // n_splits
        fold_index = split_index % n_splits

        for fraction_index, train_fraction in enumerate(fractions):
            sampled_train_idx = _sample_stratified_fraction(
                train_idx=train_idx,
                y=y,
                train_fraction=train_fraction,
                random_state=random_state + split_index + fraction_index,
            )
            predictions = _fit_and_predict(X[sampled_train_idx], y[sampled_train_idx], X[test_idx])

            fold_rows.append(
                {
                    "run": int(run_id) if run_id is not None else "",
                    "repeat": int(repeat_index),
                    "fold": int(fold_index),
                    "n_splits": int(n_splits),
                    "train_fraction": float(train_fraction),
                    "n_train_used": int(len(sampled_train_idx)),
                    "n_test": int(len(test_idx)),
                    **_compute_metrics(y[test_idx], predictions),
                }
            )

    return fold_rows


def export_evaluation_results(output_dir, cv_rows, learning_curve_rows, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cv_rows:
        cv_summary_rows = summarize_cv_rows(cv_rows)
        _write_csv(output_dir / "cv_fold_metrics.csv", cv_rows)
        _write_csv(output_dir / "cv_summary.csv", cv_summary_rows)
        _write_json(
            output_dir / "cv_summary.json",
            {
                "config": config,
                "label_mapping": PAIR_LABELS,
                "summary": cv_summary_rows,
            },
        )

    if learning_curve_rows:
        learning_curve_summary_rows = summarize_learning_curve_rows(learning_curve_rows)
        _write_csv(output_dir / "learning_curve_fold_metrics.csv", learning_curve_rows)
        _write_csv(output_dir / "learning_curve_summary.csv", learning_curve_summary_rows)
        _write_json(
            output_dir / "learning_curve_summary.json",
            {
                "config": config,
                "label_mapping": PAIR_LABELS,
                "summary": learning_curve_summary_rows,
            },
        )


def summarize_cv_rows(cv_rows):
    run_rows = _summarize_rows(cv_rows, group_keys=("run",))
    for row in run_rows:
        row["summary_level"] = "run"

    overall_rows = _summarize_rows(cv_rows, group_keys=())
    for row in overall_rows:
        row["summary_level"] = "overall"
        row["run"] = "all"

    return sorted(run_rows + overall_rows, key=lambda row: (row["summary_level"], str(row["run"])))


def summarize_learning_curve_rows(learning_curve_rows):
    run_rows = _summarize_rows(learning_curve_rows, group_keys=("run", "train_fraction"))
    for row in run_rows:
        row["summary_level"] = "run"

    overall_rows = _summarize_rows(learning_curve_rows, group_keys=("train_fraction",))
    for row in overall_rows:
        row["summary_level"] = "overall"
        row["run"] = "all"

    return sorted(
        run_rows + overall_rows,
        key=lambda row: (row["summary_level"], str(row["run"]), float(row["train_fraction"])),
    )


def _resolve_n_splits(y, requested_folds):
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        raise ValueError("Repeated stratified CV requires at least two classes.")

    max_splits = int(counts.min())
    if max_splits < 2:
        raise ValueError("Each class must have at least two samples for stratified CV.")

    return max(2, min(int(requested_folds), max_splits))


def _fit_and_predict(train_X, train_y, test_X):
    n_neighbors = max(1, min(5, len(train_y)))
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(train_X, train_y)
    return classifier.predict(test_X)


def _compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _sample_stratified_fraction(train_idx, y, train_fraction, random_state):
    if train_fraction >= 1.0:
        return np.array(train_idx, copy=True)

    rng = np.random.default_rng(random_state)
    sampled_indices = []
    train_idx = np.asarray(train_idx)
    train_labels = y[train_idx]

    for label in np.unique(train_labels):
        label_indices = train_idx[train_labels == label]
        target_size = int(np.floor(len(label_indices) * train_fraction))
        target_size = max(1, min(len(label_indices), target_size))
        sampled_indices.append(rng.choice(label_indices, size=target_size, replace=False))

    combined = np.concatenate(sampled_indices)
    rng.shuffle(combined)
    return combined


def _summarize_rows(rows, group_keys):
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[tuple(row[key] for key in group_keys)].append(row)

    summary_rows = []
    for group_values, grouped in grouped_rows.items():
        summary_row = {key: value for key, value in zip(group_keys, group_values)}
        summary_row["n_rows"] = int(len(grouped))

        for metric_name in METRIC_NAMES:
            metric_values = np.array([row[metric_name] for row in grouped], dtype=np.float64)
            summary_row[f"{metric_name}_mean"] = float(metric_values.mean())
            summary_row[f"{metric_name}_std"] = float(metric_values.std(ddof=0))

        summary_rows.append(summary_row)

    return summary_rows


def _write_csv(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
