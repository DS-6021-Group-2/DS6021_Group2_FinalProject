# Movie Potential app

A three-tab Streamlit layout for the DS 6021 movie project. Run from the repository root:

```bash
/opt/miniconda3/envs/ds6001/bin/python -m pip install -r requirements.txt
/opt/miniconda3/envs/ds6001/bin/python -m streamlit run app/app.py
```

Open the local URL printed in the terminal. Paths are resolved relative to the project, so launching from another folder with the full app path also works.

## Working now

- **Movie Predictions:** searchable movie selection and real movie details. The actual-results toggle reads saved ratings and financial labels; movies outside the financial cohort show “Unavailable.”
- **Model Comparison:** every notebook in both model folders is discovered, including SVR and KNN/PCA. Tables and charts show Original, Settings tuned, and Validation-selected scores where available. The simple models do not have a settings-only version.
- **About the Project:** sources, histories, target definitions, chronological evaluation, and limitations.

## Still to connect

The Predict button deliberately reports that models are not integrated. Prediction cards show dashes, not simulated scores. Next stage: connect saved fitted **tuned ridge** and **logistic regression** pipelines, use the validated feature schemas and selected success threshold, and limit prediction-versus-actual demonstrations to the applicable test movies. Historical results are not presented as new predictions.

## Saved results

The app uses `Data_visualization/results/{rating,success}/*.json` through the existing report validator. Reports must match the current model code, dataset, and report writer. Missing/stale reports produce an unavailable status; no scores are invented or read from stale notebook output. The reports folder is currently ignored by Git, so a fresh clone may show unavailable results until the model owners supply current reports. Creating distributable report snapshots is a later integration task; this layout does not retrain models.

Comparisons are separated by dataset/date split/test movies and outcomes. Versions are selected using validation; displayed charts use the later test period. Unknown finances are never assigned a bust label. All dataset access is read-only. The app makes no data API calls and never writes predictions, trains models, or modifies CSVs.

## Files

- `app.py`: interface, navigation, and cached read-only views.
- `model_utils.py`: movie catalog and verified result loading; no fitted models yet.

## AI assistance and verification

This layout and loader were AI-assisted. Verified using Streamlit AppTest: three-tab rendering, movie selection, Predict placeholder behavior, real actual outcomes, unavailable financial labels, chart/version controls, and missing-report fallback. Checked report metrics against saved sources and SHA-256 hashes of all dataset CSVs before/after. No commit, push, or deployment was performed.
