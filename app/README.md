# Movie Potential app

Live lasso rating predictions, three-class logistic financial predictions, all-model comparisons, and project context. Run from the repository root:

```bash
/opt/miniconda3/envs/ds6001/bin/python -m pip install -r app/requirements.txt
/opt/miniconda3/envs/ds6001/bin/python -m streamlit run app/app.py
```

## How it works

Select a later-test movie and click **Predict**. The app loads saved fitted pipelines, uses only their predictor columns, and shows all three financial probabilities and applies the hit threshold saved in the manifest. Toggle actual results to compare. Movies outside the financial cohort receive a rating prediction but financial prediction/outcome is unavailable; they are not labeled as busts. Training movies are excluded from the demo.

Lasso uses the validation-selected representation, penalty and trimmed feature list recorded in the manifest. Logistic estimates Bust / Moderate success / Hit using overlap-reviewed, validation-selected features and settings. Below the selected hit threshold, the larger Bust/Moderate probability determines the label. Preprocessing is saved inside each fitted pipeline. No training or model search happens when the app opens. Probabilities are model estimates, not calibrated guarantees.

## Models and comparisons

`artifacts/` contains two compact fitted pipelines, a provenance manifest, and metrics for all 12 implemented model notebooks. The loader checks artifact hashes, dataset/code/shared-helper hashes, and scikit-learn version before loading models. Do not load uploaded/untrusted joblib files.

Comparisons prefer current local `Data_visualization/results/` reports and fall back to packaged metrics when those ignored files are absent on a fresh clone. Both sources require matching dataset/code/writer/shared-helper hashes and the three-class outcome schema. Stale results are marked unavailable. Charts keep evaluation cohorts separate. CSVs and source notebooks are read-only inputs.

## Rebuild artifacts offline

Only needed after changing model code, selection, or datasets. First ensure all model owners have current notebook reports, then run from the repository root with full project dependencies installed:

```bash
/opt/miniconda3/envs/ds6001/bin/python app/export_models.py
```

This reruns the lasso and logistic notebooks' tagged training/selection cells in memory, fits only earlier training rows, and refuses to export unless every held-out prediction matches the notebook's saved report. It also packages comparison metrics. Use `--rating-only` to export lasso while retaining the verified success pipeline without fitting it again. It does not rewrite notebooks or CSVs. Commit updated artifacts together with their matching source code/data.

## Deploy to Streamlit Community Cloud

Use the connected account and these settings after the changes are on GitHub:

- Repository: `DS-6021-Group-2/DS6021_Group2_FinalProject`
- Branch: **Walt**
- Main file: **app/app.py**
- Python: **3.14**, matching ds6001; confirm the platform offers it.
- Dependencies: `app/requirements.txt` (matching pinned model-library versions).
- No secrets or `.env` values are required by the app.

Do not retrain models on startup. If Python 3.14 or the pinned dependencies are unavailable on the platform, stop and choose a compatible hosting environment rather than silently loading models under different library versions. This repository does not contain a public deployment URL until publishing succeeds.

## Files and verification

- `app.py`: UI and cached prediction calls.
- `model_utils.py`: read-only data, validated models, and predictions.
- `export_models.py`: explicit offline build, never imported by the app.
- `verify_models.py`: read-only inventory, score, pipeline, and Streamlit UI checks.
- `artifacts/`: portable models and verified comparison scores.

AI-assisted implementation. Tests compared all **3,871 lasso predictions** and **1,384 three-class logistic probability vectors/labels** with the notebooks after serialization/reload. Streamlit AppTest verified live Predict, selection reset, actual outcomes, missing financial data, training exclusion, and portable comparison loading without local reports. Dataset SHA-256 hashes stayed unchanged. Browser-based visual verification could not complete because browser access was unavailable.

## Verify consistency

From the repository root:

```bash
/opt/miniconda3/envs/ds6001/bin/python app/verify_models.py
```

Checks every current notebook against app inventory and packaged metrics, then verifies the loaded Lasso/logistic estimator types, notebook provenance, selected features/thresholds, held-out predictions, and live UI behavior. Full per-movie comparisons use existing local reports when available; the lasso metrics are also checked directly against the manifest. The financial pipeline has been rebuilt for all three categories; old binary artifacts and scores are rejected. UI model names and rating hyperparameter captions are read from the exported manifest to avoid stale labels.


Financial boundaries: **Bust ≤1×**, **Moderate success >1× and <2.5×**, **Hit ≥2.5×** budget. Missing financial outcomes remain unavailable. Model comparisons include macro F1 and moderate recall in addition to hit metrics; binary-era results are not mixed with current reports.

Model Comparison includes **Before overlap trim** to preserve the previous selected model. The new selected predictors/settings come from chronological training-validation comparisons, not test performance. The diagnostics guide records removed features and VIF changes; datasets retain their complete feature columns.


The exporter stops at the notebook cell tagged `model-training` and reads its `run` result. It does not depend on cell numbers or execute the results/diagnostics/report sections. Keep that tag on the training cell when reorganizing notebooks. Shared workflow code is included in report/artifact provenance checks.

The shared policy defines the common financial precision/F1 goal and is checked with every model report. Main-branch binary snapshot data is not loaded by the app.
