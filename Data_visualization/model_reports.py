"""Save checked model reports for plotting; never write dataset CSVs or fitted models."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Modeling_scripts'))
from financial_outcomes import class_probabilities, predict_classes, financial_metrics


def dependency_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): file_hash(p) for p in [
        root / 'Modeling_scripts/financial_outcomes.py', root / 'Modeling_scripts/model_diagnostics.py',
        root / 'Modeling_scripts/overlap_review.py',
        root / 'Modeling_scripts/model_workflow.py', root / 'Modeling_scripts/project_policy.py',
    ]}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def notebook_code_hash(path):
    notebook = nbformat.read(path, as_version=4)
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    return hashlib.sha256(json.dumps(sources, ensure_ascii=False).encode()).hexdigest()


def _records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=12))


def _metrics(actual, predicted, probability=None):
    if probability is None:
        return {
            "MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted)),
        }
    return financial_metrics(actual, predicted, probability)


def _validation(ns, simple, classification, variant):
    rename = {
        "Hit_F1": "Hit F1", "F1": "Hit F1", "Hit_precision": "Hit precision",
        "Precision": "Hit precision", "Hit_recall": "Hit recall", "Recall": "Hit recall",
        "Balanced_accuracy": "Balanced accuracy", "ROC_AUC": "ROC AUC", "Macro_F1": "Macro F1",
    }
    desired = {"MAE", "RMSE", "R2", "Hit F1", "Hit precision", "Hit recall", "Accuracy", "Balanced accuracy", "ROC AUC", "Macro F1"}
    if variant in ["Selected", "Before overlap trim"]:
        if variant == "Before overlap trim":
            row = ns["overlap_reference_validation"]
        elif simple:
            key = (ns["selected_version"], ns["selected_threshold"]) if classification else ns["selected_version"]
            row = ns["feature_comparison"].loc[key]
        else:
            row = ns["winner"]
        return {rename.get(key, key): float(value) for key, value in row.items()
                if rename.get(key, key) in desired and pd.notna(value)}
    folds = ns.get("feature_fold_results")
    if folds is None:
        return {}
    selected = folds.loc[folds["Version"].eq("Original director history" if simple else "All features")]
    if simple and classification:
        selected = selected.loc[selected["Threshold"].eq(ns["BASE_THRESHOLD"])]
    return {rename.get(key, key): float(selected[key].mean()) for key in selected
            if rename.get(key, key) in desired and len(selected)}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def save_model_report(ns, notebook_path):
    """Called after a complete model run. Store test predictions and train/test metrics."""
    notebook_path = Path(notebook_path).resolve()
    root = notebook_path.parents[2]
    classification = notebook_path.parent.name == "success_models"
    task = "success" if classification else "rating"
    simple = notebook_path.stem == "simple_model"
    data_path = Path(ns["DATA_FILE"]).resolve()
    movies, y_train, y_test = ns["movies"], ns["y_train"], ns["y_test"]
    assert set(y_train.index).isdisjoint(y_test.index)
    assert movies.loc[y_train.index, "release_date"].max() < movies.loc[y_test.index, "release_date"].min()
    assert set(ns["FEATURE_COLUMNS"]).isdisjoint({"success", "vote_average", "tmdb_id", "imdb_id", "title_tmdb", "release_date"})
    base_threshold = float(ns.get("BASE_THRESHOLD", 0.5))
    if simple:
        entries = {
            "Original": (ns["model"], ns["X_train"], ns["X_test"], base_threshold, ns["FEATURE_COLUMNS"]),
            "Selected": (ns["feature_selected_model"], ns["comparison_X"].loc[y_train.index], ns["comparison_X"].loc[y_test.index], float(ns.get("selected_threshold", base_threshold)), ns["selected_columns"]),
        }
    else:
        entries = {
            "Original": (ns["initial_model"], ns["X_train"], ns["X_test"], base_threshold, ns["FEATURE_COLUMNS"]),
            "Settings only": (ns["settings_model"], ns["X_train"], ns["X_test"], base_threshold, ns["FEATURE_COLUMNS"]),
            "Selected": (ns["model"], ns["X_train"], ns["X_test"], float(ns.get("selected_threshold", base_threshold)), ns["selected_feature_columns"]),
        }
    if 'overlap_reference_model' in ns:
        entries['Before overlap trim'] = (ns['overlap_reference_model'], ns['X_train'], ns['X_test'],
            float(ns['overlap_reference_threshold']) if classification else base_threshold, ns['overlap_reference_features'])
    metadata = movies.loc[y_test.index, ["tmdb_id", "title_tmdb", "release_date"]].copy()
    metadata["actual"] = y_test.to_numpy()
    versions = {}
    for variant, (estimator, train_x, test_x, threshold, features) in entries.items():
        if classification:
            train_probability = class_probabilities(estimator, train_x)
            test_probability = class_probabilities(estimator, test_x)
            train_prediction = predict_classes(train_probability, threshold)
            test_prediction = predict_classes(test_probability, threshold)
        else:
            train_probability = test_probability = None
            train_prediction = estimator.predict(train_x)
            test_prediction = estimator.predict(test_x)
        assert np.isfinite(test_prediction).all()
        predictions = metadata.copy()
        predictions["predicted"] = test_prediction
        if classification:
            for i, label in enumerate(['bust_probability', 'moderate_probability', 'hit_probability']):
                predictions[label] = test_probability[:, i]
        else:
            predictions["residual"] = y_test.to_numpy() - test_prediction
        versions[variant] = {
            "threshold": threshold if classification else None,
            "features": list(features),
            "train_metrics": _metrics(y_train, train_prediction, train_probability),
            "test_metrics": _metrics(y_test, test_prediction, test_probability),
            "validation": _validation(ns, simple, classification, variant) if variant != "Settings only" else {},
            "predictions": _records(predictions),
        }
    baseline = ns["baseline"]
    dummy_x = np.zeros((len(y_test), 1))
    baseline_prediction = baseline.predict(dummy_x)
    baseline_probability = class_probabilities(baseline, dummy_x) if classification else None
    report = {
        "schema_version": 2,
        "outcome_schema": "bust-moderate-hit-v1" if classification else "rating-v1",
        "dependencies_sha256": dependency_hashes(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": task, "model_key": notebook_path.stem,
        "notebook": str(notebook_path.relative_to(root)),
        "code_sha256": notebook_code_hash(notebook_path),
        "writer_sha256": file_hash(__file__),
        "dataset": str(data_path.relative_to(root)), "dataset_sha256": file_hash(data_path),
        "cutoff": ns["CUTOFF_DATE"], "train_count": len(y_train), "test_count": len(y_test),
        "model_settings": ns["MODEL_SETTINGS"], "parameter_grid": ns.get("PARAM_GRID", {}),
        "selection_metric": ns.get("TUNING_METRIC", "Hit_F1" if classification else "MAE"),
        "precision_goal": ns.get("MIN_VALIDATION_PRECISION") if classification else None,
        "precision_requirement_met": bool(ns.get("precision_requirement_met", False)) if classification else None,
        "baseline_test_metrics": _metrics(y_test, baseline_prediction, baseline_probability),
        "versions": versions,
        "feature_fold_results": _records(ns["feature_fold_results"]),
        "overlap_review": {
            "status": ns['overlap_review']['status'], "rules": ns['overlap_review']['rules'],
            "comparison": _records(ns['diagnostic_results']['overlap_comparison']),
            "candidate_count": len(ns['overlap_review']['candidates']),
        } if 'overlap_review' in ns else None,
        "overlap_fold_results": _records(ns['overlap_review']['folds']) if 'overlap_review' in ns else [],
    }
    directory = Path(__file__).resolve().parent / "results" / task
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{notebook_path.stem}.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default) + "\n")
    temporary.replace(output)
    return output


def load_reports(task, model_keys, variant="Selected"):
    """Load reports only when their dataset, model code, and report writer still match."""
    root = Path(__file__).resolve().parents[1]
    reports = {}
    reference = None
    signature = None
    for key in model_keys:
        path = Path(__file__).resolve().parent / "results" / task / f"{key}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing report: {path.name}. Save and Run All in Modeling_scripts/{task}_models/{key}.ipynb first.")
        report = json.loads(path.read_text())
        for label, current, stored in [
            ("dataset", file_hash(root / report["dataset"]), report["dataset_sha256"]),
            ("notebook code", notebook_code_hash(root / report["notebook"]), report["code_sha256"]),
            ("report writer", file_hash(__file__), report["writer_sha256"]),
        ]:
            if current != stored:
                raise ValueError(f"Stale {label} for {report['notebook']}. Save that notebook and Run All to refresh its visualization report.")
        if report.get('dependencies_sha256') != dependency_hashes():
            raise ValueError(f"Stale shared model helpers for {report['notebook']}. Run All to refresh.")
        if task == 'success' and report.get('outcome_schema') != 'bust-moderate-hit-v1':
            raise ValueError('A binary report cannot be used for the three-class task.')
        if variant not in report["versions"]:
            raise ValueError(f"Variant {variant!r} is unavailable in {report['notebook']}.")
        current_signature = (report["dataset_sha256"], report["cutoff"], report["train_count"], report["test_count"])
        predictions = pd.DataFrame(report["versions"][variant]["predictions"])
        assert predictions["tmdb_id"].is_unique
        cohort = predictions.set_index("tmdb_id")[["actual", "release_date"]].sort_index()
        if signature is None:
            signature, reference = current_signature, cohort
        else:
            if signature != current_signature:
                raise ValueError("Reports have different datasets or date splits. Use matching settings before comparing models.")
            pd.testing.assert_frame_equal(reference, cohort, check_dtype=False)
        reports[key] = report
    return reports
