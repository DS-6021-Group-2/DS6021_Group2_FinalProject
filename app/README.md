# Movie Potential app

Live ridge rating predictions, logistic hit/bust predictions, all-model comparisons, and project context. Run from the repository root:

```bash
/opt/miniconda3/envs/ds6001/bin/python -m pip install -r app/requirements.txt
/opt/miniconda3/envs/ds6001/bin/python -m streamlit run app/app.py
```

## How it works

Select a later-test movie and click **Predict**. The app loads saved fitted pipelines, uses only their predictor columns, and applies the selected logistic **0.65 threshold**. Toggle actual results to compare. Movies outside the financial cohort receive a rating prediction but financial prediction/outcome is unavailable; they are not labeled as busts. Training movies are excluded from the demo.

Ridge uses quadratic numeric terms, alpha 100, and movie/director/actor features. Logistic uses its original estimator settings and selected movie/director features. Preprocessing is saved inside each fitted pipeline. No training or model search happens when the app opens. Probabilities are model estimates, not calibrated guarantees.

## Models and comparisons

`artifacts/` contains two compact fitted pipelines, a provenance manifest, and metrics for all 11 implemented model notebooks. The loader checks artifact hashes, dataset/code hashes, and scikit-learn version before loading models. Do not load uploaded/untrusted joblib files.

Comparisons prefer current local `Data_visualization/results/` reports and fall back to packaged metrics when those ignored files are absent on a fresh clone. Both sources require matching dataset/code/writer hashes. Stale results are marked unavailable. Charts keep evaluation cohorts separate. CSVs and source notebooks are read-only inputs.

## Rebuild artifacts offline

Only needed after changing model code, selection, or datasets. First ensure all model owners have current notebook reports, then run from the repository root with full project dependencies installed:

```bash
/opt/miniconda3/envs/ds6001/bin/python app/export_models.py
```

This reruns the two notebooks' existing training/selection cells in memory, fits only earlier training rows, and refuses to export unless every held-out prediction matches the notebook's saved report. It also packages comparison metrics. It does not rewrite notebooks or CSVs. Commit updated artifacts together with their matching source code/data.

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
- `artifacts/`: portable models and verified comparison scores.

AI-assisted implementation. Tests compared all **3,871 ridge predictions** and **1,384 logistic probabilities/labels** with the notebooks after serialization/reload. Streamlit AppTest verified live Predict, selection reset, actual outcomes, missing financial data, training exclusion, and portable comparison loading without local reports. Dataset SHA-256 hashes stayed unchanged. Browser-based visual verification could not complete because browser access was unavailable.
