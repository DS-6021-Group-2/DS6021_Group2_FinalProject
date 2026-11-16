# Shared modeling code

The notebooks, app, and verification scripts import these helpers automatically. Run and tune models from the notebooks.

| File | Purpose |
|---|---|
| `model_workflow.py` | Data checks, chronological tuning, evaluation, and reports. |
| `project_policy.py` | Shared dates, validation goals, and history feature definitions. |
| `financial_outcomes.py` | Bust/moderate/hit labels, probabilities, and metrics. |
| `model_diagnostics.py` | VIF, numerical stability, and model quality checks. |
| `overlap_review.py` | Validation-based feature trimming and exact redundancy removal. |
| `director_history_comparison.py` | Models with versus without director history. |

Verification commands are in [tests](../../tests/README.md); model instructions are in the [modeling guide](../README.md).
