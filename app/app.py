"""Streamlit layout for the DS 6021 project. AI-assisted; see README.md."""
from pathlib import Path
import sys
import json

import pandas as pd
import streamlit as st

# Use the project package instead of a generic module cached by an older app run.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Reload helpers each run so a Cloud auto-update cannot keep an older copy in memory.
import importlib
import app.model_utils
importlib.reload(app.model_utils)
from app.model_utils import CLASS_LABELS, SUCCESS_MIN_VALIDATION_PRECISION, ROOT, LABELS, load_movie_catalog, load_comparisons, load_prediction_models, predict_movie

st.set_page_config(page_title='Movie Potential | DS 6021', page_icon='🎬', layout='wide')
st.markdown('''<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
.hero {background: linear-gradient(115deg,#102b35,#174e56); color:#fff;
       padding:2rem 2.3rem; border-radius:18px; margin-bottom:1.4rem;}
.hero .eyebrow {color:#a9e0d8; letter-spacing:.16em; font-size:.75rem; font-weight:700;}
.hero h1 {color:#fff; font-size:2.7rem; margin:.35rem 0; padding:0;}
.hero p {color:#d2e5e7; max-width:680px; margin:.4rem 0;}
div[data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;}
button[data-baseweb="tab"] {font-weight:600;}
</style>
<div class="hero"><div class="eyebrow">DS 6021 · GROUP 2</div>
<h1>Movie Potential</h1><p>Two perspectives on a movie: how viewers rate it, and how it performs at the box office.</p></div>''', unsafe_allow_html=True)

# Cache by file modification signatures so existing results refresh without retraining.
def input_signature():
    paths = [ROOT / 'app/model_utils.py', ROOT / 'data/model_data_rating.csv', ROOT / 'data/model_data_success.csv',
             ROOT / 'Data_visualization/model_reports.py', ROOT / 'Modeling_scripts/helpers/financial_outcomes.py',
             ROOT / 'Modeling_scripts/helpers/model_diagnostics.py', ROOT / 'Modeling_scripts/helpers/overlap_review.py',
             ROOT / 'Modeling_scripts/helpers/model_workflow.py', ROOT / 'Modeling_scripts/helpers/project_policy.py']
    paths += list((ROOT / 'app/artifacts').glob('*'))
    paths += list((ROOT / 'Modeling_scripts').glob('*/*.ipynb'))
    paths += list((ROOT / 'Data_visualization/results').glob('*/*.json'))
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(paths) if p.exists())

@st.cache_data(show_spinner=False)
def read_inputs(signature):
    return load_movie_catalog(), load_comparisons()

try:
    catalog, (scores, statuses, cohorts) = read_inputs(input_signature())
    prediction_metadata = json.loads((ROOT / 'app/artifacts/manifest.json').read_text())['models']
    rating_meta = prediction_metadata['rating']
    rating_key = rating_meta.get('model_key', Path(rating_meta['notebook']).stem)
    rating_name = LABELS.get(rating_key, rating_key.replace('_', ' ').title())
    rating_params = rating_meta['selected_parameters']
    degree = rating_params.get('preprocessor__numeric__polynomial__degree', 1)
    rating_details = 'Quadratic features' if degree == 2 else 'Linear features'
    if 'regression__alpha' in rating_params:
        rating_details += f" · Alpha {rating_params['regression__alpha']:g}"

except (OSError, ValueError, AssertionError, KeyError) as exc:
    st.error(f'Project data could not be loaded ({type(exc).__name__}). Check the saved model CSVs and app README.')
    st.stop()

@st.cache_resource(show_spinner=False)
def read_models(signature):
    return load_prediction_models()

with st.sidebar:
    st.title('Movie Potential')
    st.caption('A movie prediction research project')
    st.success('Live model predictions · later-test movies only.')
    st.markdown(f'**Prediction models**\n\nRatings → {rating_name}\n\nBust / moderate / hit → logistic regression')
    st.caption('Models are fitted offline. Opening the app never trains them.')

summary = st.columns(3)
summary[0].metric('Rated test movies', f'{len(catalog):,}')
summary[1].metric('Finance-eligible test movies', f'{catalog.success.notna().sum():,}')
summary[2].metric('Models in comparison', str(len(statuses)))
predict_tab, compare_tab, about_tab = st.tabs(['Movie Predictions', 'Model Comparison', 'About the Project'])

with predict_tab:
    st.subheader('Explore a movie')
    st.caption('Search by title or TMDB ID. Choose a held-out movie to generate real model predictions, then reveal its actual outcomes.')
    labels = catalog.set_index('tmdb_id')['label'].to_dict()
    movie_id = st.selectbox('Choose a movie', options=catalog.tmdb_id.tolist(),
                            format_func=labels.get, index=None, placeholder='Start typing a movie title…', key='movie')
    clicked = st.button('Predict', type='primary', disabled=movie_id is None)
    if clicked:
        try:
            with st.spinner('Generating predictions…'):
                prediction = predict_movie(movie_id, read_models(input_signature()))
            st.session_state['movie_prediction'] = (movie_id, input_signature(), prediction)
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            st.session_state.pop('movie_prediction', None)
            st.error(f'Predictions unavailable: {exc}')
    stored = st.session_state.get('movie_prediction')
    prediction = stored[2] if stored and stored[0] == movie_id and stored[1] == input_signature() else None
    if movie_id is not None:
        movie = catalog.loc[catalog.tmdb_id.eq(movie_id)].iloc[0]
        st.subheader(movie.title_tmdb)
        runtime = f'{movie.runtime:g} min' if pd.notna(movie.runtime) else 'Runtime unavailable'
        st.caption(f"Released {movie.release_date:%B %d, %Y} · {runtime}")

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.caption('VIEWER RESPONSE')
        st.metric('Predicted viewer rating', f"{prediction['rating']['rating']:.2f} / 10" if prediction else '—')
        st.caption(f'{rating_name} · {rating_details}')
    with right, st.container(border=True):
        st.caption('BOX-OFFICE PERFORMANCE')
        financial = prediction.get('success') if prediction else None
        st.metric('Estimated hit probability', f"{financial['probability']:.1%}" if financial and 'probability' in financial else '—')
        if financial and 'probability' in financial:
            st.markdown(f"**Predicted outcome: {financial['label']}**")
            st.write(' · '.join(f'{label}: {probability:.1%}' for label, probability in financial['probabilities'].items()))
            st.caption(f"Three-class logistic regression · Hit threshold {financial['threshold']:.0%}. Model probabilities are not guarantees.")
            st.caption('Below the hit threshold, the larger Bust or Moderate success probability determines the label.')
        elif financial:
            st.caption(financial['unavailable'])
        else:
            st.caption('Logistic regression · Select a movie and click Predict.')
    show_actual = st.toggle('Show actual results', key='show_actual')
    if show_actual and movie_id is None:
        st.info('Choose a movie to view its actual outcomes.')
    elif show_actual:
        a, b = st.columns(2)
        a.metric('Actual TMDB viewer rating', f'{movie.vote_average:.2f} / 10')
        outcome = 'Unavailable' if pd.isna(movie.success) else CLASS_LABELS[int(movie.success)]
        b.metric('Actual financial outcome', outcome)
        if pd.isna(movie.success):
            st.caption('This movie is outside the financially eligible cohort. Missing or insufficient financial data is not labeled as a bust.')
    st.caption('Project benchmark: Bust: revenue/budget ≤1. Moderate success: 1<ratio<2.5. Hit: ratio ≥2.5. This is a revenue-to-budget label, not audited profitability.')

with compare_tab:
    st.subheader('How do the models compare?')
    st.write('Explore every implemented model, including KNN with PCA. These are saved later-test scores, not live predictions.')
    st.info('Compare models only on matching datasets and test periods. Each chart below is restricted to one verified evaluation cohort. Validation selects settings; later-test scores describe performance.')
    for task, title in [('rating', 'Viewer rating models'), ('success', 'Financial success models')]:
        st.markdown(f'### {title}')
        task_scores = scores.loc[scores.Task.eq(task)].copy() if not scores.empty else pd.DataFrame()
        task_status = statuses.loc[statuses.Task.eq(task)].copy()
        unavailable = task_status.loc[~task_status.Status.eq('Verified saved results')]
        if task_scores.empty:
            st.warning('Saved results are unavailable. Models are listed below; no scores have been filled in.')
            st.dataframe(task_status[['Model', 'Status']], hide_index=True, width='stretch')
            continue
        cohort_ids = task_scores.Cohort.drop_duplicates().tolist()
        if len(cohort_ids) > 1:
            st.warning('Multiple evaluation cohorts found. Choose one; results from different cohorts are not combined.')
        cohort = st.selectbox('Evaluation cohort', cohort_ids, key=f'{task}_cohort',
                              format_func=lambda k, current_task=task: f"{cohorts[(current_task,k)]['Test movies']:,} test movies · from {cohorts[(current_task,k)]['Test starts']} · {k}")
        details = cohorts[(task, cohort)]
        st.caption(f"{details['Dataset']} · {details['Training movies']:,} training / {details['Test movies']:,} test movies · test starts {details['Test starts']}")
        in_cohort = task_scores.loc[task_scores.Cohort.eq(cohort)]
        baseline_name = 'Baseline (predicts mean rating)' if task == 'rating' else 'Baseline (predicts most common class)'
        tuned = in_cohort.loc[in_cohort.Version.eq('Validation-selected')].assign(Type='Tuned model')
        baseline = in_cohort.loc[in_cohort.Version.eq('Baseline')].head(1).assign(Model=baseline_name, Type='Baseline')
        filtered = pd.concat([baseline, tuned], ignore_index=True)
        metrics = ['MAE', 'RMSE', 'R2'] if task == 'rating' else ['Macro F1', 'Moderate recall', 'Moderate F1', 'Hit precision', 'Hit recall', 'Hit F1', 'Accuracy', 'Balanced accuracy', 'ROC AUC', 'Multiclass ROC AUC']
        if tuned.empty:
            st.info('No tuned-model results are saved for this cohort.')
        else:
            columns = ['Model'] + (['Threshold'] if task == 'success' else []) + metrics
            display = filtered[columns].rename(columns={'R2': 'R²'})
            st.dataframe(display, hide_index=True, width='stretch', column_config={c: st.column_config.NumberColumn(format='%.3f') for c in display if c != 'Model'})
            metric = st.selectbox('Chart metric', metrics, key=f'{task}_metric', format_func=lambda x: 'R²' if x == 'R2' else x)
            st.bar_chart(filtered, x='Model', y=metric, color='Type', horizontal=True,
                         x_label='Model', y_label='R²' if metric == 'R2' else metric)
        st.caption('Baseline = a model that ignores every feature and always predicts the training mean rating (ratings) or the most common class (success). Tuned model = the final validation-selected version of each model (and hit threshold for success).')
        if not unavailable.empty:
            st.warning('Some models have unavailable results; they remain listed here.')
            st.dataframe(unavailable[['Model', 'Status']], hide_index=True, width='stretch')
        with st.expander(f'What do the {task} metrics mean?'):
            if task == 'rating':
                st.markdown('**MAE ↓:** average rating error in points. **RMSE ↓:** gives larger errors more weight. **R² ↑:** variation explained relative to the test-set mean; it can be negative.')
            else:
                st.markdown('**Hit precision:** how often predicted hits are actual hits. **Recall:** how many actual hits are found. **F1:** balances precision and recall. **Accuracy:** fraction of all correct labels. **Macro F1:** equally averages all three class F1 scores. **Moderate recall:** fraction of actual moderate movies identified. **ROC AUC:** Hit vs rest ranking. **Multiclass ROC AUC:** macro average of three one-vs-rest rankings.')
                st.caption(f'The selection goal is ≥{SUCCESS_MIN_VALIDATION_PRECISION:.0%} mean validation hit precision, then highest hit F1. It does not guarantee {SUCCESS_MIN_VALIDATION_PRECISION:.0%} test precision. The hit threshold selects Hit when reached; otherwise the larger Bust/Moderate probability wins. Inspect moderate recall and macro F1 alongside hit metrics.')
    with st.expander('Model inventory and result availability'):
        st.dataframe(statuses, hide_index=True, width='stretch')

with about_tab:
    st.subheader('Two questions. One movie project.')
    st.write('How accurately can movie characteristics and a director’s earlier film history predict viewer ratings and financial success?')
    a, b = st.columns(2)
    with a, st.container(border=True):
        st.markdown('#### Viewer ratings')
        st.write('Predict TMDB vote_average on a 0–10 scale. Rated movies remain eligible even without usable budgets or revenues.')
    with b, st.container(border=True):
        st.markdown('#### Financial success')
        st.write('Predict three categories: Bust (ratio ≤1), Moderate success (1<ratio<2.5), and Hit (ratio ≥2.5). The financial cohort requires reported budget ≥ $250,000 and revenue ≥ $1,000. Hit means revenue ≥ 2.5 × budget.')
    st.markdown('### Where the information comes from')
    st.write('TMDB supplies movie characteristics, ratings, budgets, revenues, cast and companies. IMDb supplies identities and earlier director film histories; an IMDb actor supplement improves some early five-year windows.')
    st.markdown('### From raw data to predictions')
    st.write('Collect → combine → clean → engineer earlier-film histories → chronological validation → evaluate later movies.')
    st.write('Features include genres, runtime, release timing, log budget, smoothed earlier director success and ratings, experience counts, five-year actor/company activity, and franchise history. Current ratings and financial success are outcomes, never predictor inputs.')
    st.write('Models use a November 18, 2017 cutoff: train before it and test on or after it. Three expanding validation folds stay within training. Imputation, encoding, scaling and PCA, where used, are fitted only on each training fold.')
    with st.expander('Limitations and interpretation', expanded=True):
        st.markdown('''- Unknown budgets remain missing in rating data and are imputed from training data. Missing financial outcomes are not busts.
- The sample favors sufficiently voted movies; histories are incomplete and use retrospective snapshots.
- Earlier histories update through time: this is rolling evaluation, not one frozen forecast at the split date.
- Repeated test inspection makes findings exploratory. Model scores and hit likelihoods are not guarantees.
- PCA was tested with KNN; clustering is not implemented in this layout.
- Predictions use saved lasso and logistic pipelines fitted only on earlier training movies. The demo includes later-test movies only; financial predictions are unavailable outside the eligible financial cohort.''')
    st.caption('DS 6021 · Group 2 · AI-assisted app layout; implementation and verification notes are in app/README.md.')
