# Verification checks

These optional scripts check the project; normal training and app startup do not require running them. Run from the project root using `ds6001`.

| Script | Checks |
|---|---|
| `verify_diagnostics.py` | VIF calculations, exact dependencies and constants. |
| `verify_financial_outcomes.py` | Bust/moderate/hit boundaries, metrics and dataset agreement. |
| `verify_model_workflow.py` | Chronological splits, input safeguards and test-outcome isolation. |
| `verify_overlap_review.py` | Trimming tolerances, precision guards and exact-alias removal. |
| `verify_project_consistency.py` | Shared labels, goals, dates and model inputs. |

```bash
/opt/miniconda3/envs/ds6001/bin/python tests/verify_diagnostics.py
/opt/miniconda3/envs/ds6001/bin/python tests/verify_financial_outcomes.py
/opt/miniconda3/envs/ds6001/bin/python tests/verify_model_workflow.py
/opt/miniconda3/envs/ds6001/bin/python tests/verify_overlap_review.py
/opt/miniconda3/envs/ds6001/bin/python tests/verify_project_consistency.py --artifacts
```

The workflow check fits small synthetic models in a temporary directory. The artifact check reads existing reports or packaged app results; it does not retrain project models. No script rewrites project CSVs.

The separate app/UI check remains at `app/verify_models.py`.
