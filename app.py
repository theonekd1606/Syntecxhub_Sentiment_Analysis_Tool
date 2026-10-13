"""
app.py - the Streamlit interface for Sentiment AI.

Start it with:  streamlit run app.py

The app never retrains on its own. It loads the saved model from models/ (cached, so
Streamlit reruns are instant). Training happens through train_model.py, or through the
"Train on a dataset" panel in the Dataset tab.
"""

import html
import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import train_model as tm
from preprocess import LABELS, MAX_TEXT_LENGTH, DatasetError, detect_columns, load_dataset, read_csv_safely

st.set_page_config(page_title="Sentiment AI", page_icon="🟠", layout="wide")

# Tone for each class: (colour, display name)
TONES = {"positive": ("#3ddc84", "Positive"),
         "negative": ("#ff5a5f", "Negative"),
         "neutral": ("#f5c542", "Neutral")}

EXAMPLES = {
    "Praise": "This product is absolutely amazing!",
    "Complaint": "I had a terrible experience with this service.",
    "Plain fact": "The package arrived this morning.",
    "Twist": "I thought it would be bad, but I actually loved it.",
    "Negation": "I do not recommend this product.",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&family=DM+Sans:wght@400;500;700&display=swap');
:root{--bg:#080808;--panel:#141414;--panel2:#1c1c1c;--line:#2a2a2a;--orange:#ff7a1a;
      --text:#f5f5f5;--muted:#9a9a9a;}
.stApp{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1180px;padding-top:2.2rem;padding-bottom:4rem;}
h1,h2,h3,h4{font-family:'Sora',sans-serif;letter-spacing:-0.01em;color:var(--text);}
p,li,label,span{font-family:'DM Sans',sans-serif;}

/* ---- hero ---- */
.hero{border-left:6px solid var(--orange);padding:.2rem 0 .2rem 1.2rem;margin-bottom:1.4rem;}
.brand{font-family:'Sora',sans-serif;font-weight:800;font-size:clamp(2.3rem,5.5vw,3.8rem);
       letter-spacing:-0.03em;line-height:1.05;margin:0;color:var(--text);}
.tagline{font-size:1.2rem;color:var(--orange);margin:.45rem 0 .6rem;font-weight:500;}
.blurb{color:var(--muted);max-width:68ch;line-height:1.6;margin:0;}
.pill{display:inline-block;margin:.9rem .5rem 0 0;padding:.28rem .8rem;border-radius:999px;
      border:1px solid var(--line);background:var(--panel);color:var(--text);font-size:.85rem;}
.pill.warn{border-color:var(--orange);color:var(--orange);}
.pill i{display:inline-block;width:8px;height:8px;border-radius:50%;background:#3ddc84;margin-right:.45rem;}
.pill.warn i{background:var(--orange);}

/* ---- tabs ---- */
button[data-baseweb="tab"]{color:var(--muted);font-family:'Sora',sans-serif;font-weight:600;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--orange);}
div[data-baseweb="tab-highlight"]{background:var(--orange);}
div[data-baseweb="tab-border"]{background:var(--line);}

/* ---- inputs ---- */
div[data-baseweb="textarea"]{background:#101010;border-radius:12px;border:1px solid var(--line);}
div[data-baseweb="textarea"]:focus-within{border-color:var(--orange);}
[data-testid="stTextArea"] textarea{background:#101010;color:var(--text);font-size:1.05rem;line-height:1.55;}

/* ---- buttons ---- */
.stButton>button,.stDownloadButton>button{background:var(--panel2);color:var(--text);
  border:1px solid var(--line);border-radius:10px;font-weight:500;transition:border-color .15s,color .15s,transform .15s;}
.stButton>button:hover{border-color:var(--orange);color:var(--orange);}
.stButton>button:focus-visible{outline:2px solid var(--orange);outline-offset:2px;}
.stButton>button[kind="primary"],.stButton>button[data-testid="stBaseButton-primary"]{
  background:var(--orange);border-color:var(--orange);color:#0a0a0a;font-weight:700;}
.stButton>button[kind="primary"]:hover,.stButton>button[data-testid="stBaseButton-primary"]:hover{
  background:#ff9248;color:#0a0a0a;transform:translateY(-1px);}

/* ---- cards ---- */
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1.3rem 1.4rem;
      box-shadow:0 10px 30px rgba(0,0,0,.45);}
.card h4{margin:0 0 .5rem;font-size:1.05rem;}
.card p{color:var(--muted);line-height:1.6;margin:.2rem 0;}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:.4rem 0 1.2rem;}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1rem 1.1rem;
      box-shadow:0 8px 22px rgba(0,0,0,.4);transition:border-color .15s;}
.stat:hover{border-color:var(--orange);}
.stat b{display:block;font-family:'Sora',sans-serif;font-size:1.8rem;font-weight:700;color:var(--text);}
.stat span{color:var(--muted);font-size:.9rem;}
.stat.tone b{color:var(--tone);}

/* ---- result ---- */
.result{background:var(--panel);border:1px solid var(--line);border-top:4px solid var(--tone);
        border-radius:14px;padding:1.4rem 1.5rem;box-shadow:0 10px 30px rgba(0,0,0,.45);}
.result-head{display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:.4rem;}
.result-label{font-family:'Sora',sans-serif;font-weight:800;font-size:2.3rem;color:var(--tone);line-height:1.1;}
.result-conf{color:var(--muted);}
.result-conf b{color:var(--text);}
.prob{display:grid;grid-template-columns:78px 1fr 54px;align-items:center;gap:.7rem;margin:.55rem 0;}
.prob-name{color:var(--muted);font-size:.92rem;}
.prob-val{text-align:right;font-variant-numeric:tabular-nums;}
.track{background:#0d0d0d;border:1px solid var(--line);border-radius:999px;height:12px;overflow:hidden;}
.fill{height:100%;border-radius:999px;transform-origin:left;animation:grow .6s ease-out;}
@keyframes grow{from{transform:scaleX(0);}to{transform:scaleX(1);}}
.chips{margin:.4rem 0 .2rem;}
.chip{display:inline-block;margin:.2rem .35rem .2rem 0;padding:.18rem .65rem;border-radius:8px;
      background:var(--panel2);border:1px solid var(--line);font-size:.9rem;}
.fine{color:var(--muted);font-size:.85rem;line-height:1.5;}
.empty{border:1px dashed var(--line);border-radius:14px;padding:2.2rem 1.4rem;color:var(--muted);text-align:center;}
@media (prefers-reduced-motion:reduce){.fill{animation:none;}.stat,.stButton>button{transition:none;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner=False)
def _load(model_mtime: float):
    """Cached model loading. The file's modification time is part of the cache key,
    so retraining automatically refreshes the cache."""
    return tm.load_artifacts()


def get_artifacts():
    """Return (artifacts, error_message). Exactly one of them is None."""
    if not tm.artifacts_exist():
        return None, ("No trained model found. Train one first by running "
                      "`python train_model.py` in the project folder, or use the buttons below.")
    try:
        return _load(tm.MODEL_PATH.stat().st_mtime + tm.METRICS_PATH.stat().st_mtime), None
    except Exception as exc:  # corrupt / incompatible files
        return None, f"The saved model could not be loaded ({exc}). Retrain it with `python train_model.py`."


def chart_style(ax):
    ax.set_facecolor("#141414")
    for spine in ax.spines.values():
        spine.set_color("#2a2a2a")
    ax.tick_params(colors="#9a9a9a")


def distribution_figure(counts: dict):
    labels = [l for l in LABELS if counts.get(l, 0) > 0 or l in counts]
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#141414")
    bars = ax.bar([TONES[l][1] for l in labels], [counts[l] for l in labels],
                  color=[TONES[l][0] for l in labels], width=0.55)
    ax.bar_label(bars, color="#f5f5f5", padding=3)
    ax.set_ylabel("Records", color="#9a9a9a")
    ax.grid(axis="y", color="#2a2a2a", linewidth=0.6)
    ax.set_axisbelow(True)
    chart_style(ax)
    fig.tight_layout()
    return fig


def stat_cards(items):
    """items: list of (value, label, optional_colour)."""
    cells = ""
    for item in items:
        value, label = item[0], item[1]
        tone = f'tone" style="--tone:{item[2]}' if len(item) > 2 else ""
        cells += f'<div class="stat {tone}"><b>{html.escape(str(value))}</b><span>{html.escape(label)}</span></div>'
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


def pct(x):
    return f"{x * 100:.1f}%"


def train_and_report(source, **kwargs):
    """Train with a spinner, show friendly errors, refresh the cache on success."""
    try:
        with st.spinner("Training the model - this can take a moment for large datasets..."):
            tm.train(source, log=lambda *_: None, **kwargs)
        _load.clear()
        st.success("Training finished. The new model and its real test scores are now loaded.")
        st.rerun()
    except DatasetError as exc:
        st.error(f"Dataset problem: {exc}")
    except Exception as exc:
        st.error(f"Training failed unexpectedly: {exc}")


# ------------------------------------------------------------------ state & hero
artifacts, load_error = get_artifacts()
st.session_state.setdefault("text_input", "")
st.session_state.setdefault("result", None)


def set_example(text):
    st.session_state.text_input = text
    st.session_state.result = None


def clear_text():
    st.session_state.text_input = ""
    st.session_state.result = None


if artifacts:
    m = artifacts["metrics"]
    pills = (f'<span class="pill"><i></i>Model ready: {html.escape(m["model_name"])}</span>'
             f'<span class="pill">Trained on {m["dataset"]["total"]:,} records</span>')
    if m["dataset"]["is_sample"]:
        pills += '<span class="pill warn"><i></i>Demo dataset - not for real accuracy claims</span>'
else:
    pills = '<span class="pill warn"><i></i>No model trained yet</span>'

st.markdown(f"""
<div class="hero">
  <h1 class="brand">SENTIMENT AI</h1>
  <div class="tagline">Understand the emotion behind every word.</div>
  <p class="blurb">This application uses machine learning and natural language processing to read
  a piece of text and decide whether its sentiment is positive, negative or neutral. The model
  is trained on labelled examples, so every prediction comes from patterns it learned from data.</p>
  {pills}
</div>
""", unsafe_allow_html=True)

tab_analyze, tab_data, tab_perf, tab_about = st.tabs(
    ["Text Analysis", "Dataset", "Model Performance", "About the Project"])

# ------------------------------------------------------------------ A + B: analysis
with tab_analyze:
    if load_error:
        st.error(load_error)
        if tm.SAMPLE_DATASET.exists() and st.button("Train a demo model on the bundled sample dataset"):
            train_and_report(tm.SAMPLE_DATASET)
    elif artifacts.get("version_note"):
        st.warning(artifacts["version_note"])

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.subheader("Analyse a text")
        st.text_area("Text to analyse", key="text_input", height=190, max_chars=MAX_TEXT_LENGTH,
                     placeholder="Paste a review, a tweet or any sentence...",
                     label_visibility="collapsed")
        text = st.session_state.text_input
        st.caption(f"{len(text):,} characters - {len(text.split()):,} words "
                   f"(limit {MAX_TEXT_LENGTH:,} characters)")
        b1, b2, _ = st.columns([1.3, 1, 1.4])
        analyze = b1.button("Analyze Sentiment", type="primary", width="stretch",
                            disabled=artifacts is None)
        b2.button("Clear", on_click=clear_text, width="stretch")

        st.markdown('<p class="fine" style="margin:.9rem 0 .3rem">Try an example:</p>', unsafe_allow_html=True)
        cols = st.columns(len(EXAMPLES))
        for col, (name, sample) in zip(cols, EXAMPLES.items()):
            col.button(name, key=f"ex_{name}", on_click=set_example, args=(sample,),
                       width="stretch", help=sample)

        if analyze:
            try:
                st.session_state.result = {"text": text, **tm.predict_text(text, artifacts)}
            except ValueError as exc:
                st.session_state.result = None
                st.warning(str(exc))
            except Exception as exc:
                st.session_state.result = None
                st.error(f"Something went wrong while predicting: {exc}")

    with right:
        st.subheader("Sentiment result")
        res = st.session_state.result
        if not res:
            st.markdown('<div class="empty">Enter some text and press <b>Analyze Sentiment</b>.<br>'
                        'Or click one of the examples to load it.</div>', unsafe_allow_html=True)
        else:
            color, name = TONES.get(res["label"], ("#ff7a1a", res["label"].title()))
            conf = res["confidence"]
            conf_html = (f'<span class="result-conf"><b>{pct(conf)}</b> estimated probability</span>'
                         if conf is not None else "")
            bars = ""
            for label in LABELS:
                if label in res["probabilities"]:
                    p = res["probabilities"][label]
                    c = TONES[label][0]
                    bars += (f'<div class="prob"><span class="prob-name">{TONES[label][1]}</span>'
                             f'<div class="track"><div class="fill" style="width:{p*100:.1f}%;background:{c}"></div></div>'
                             f'<span class="prob-val">{pct(p)}</span></div>')
            chips = "".join(f'<span class="chip">{html.escape(t)}</span>' for t in res["top_terms"])
            low = ""
            if conf is not None and conf < 0.5:
                low = ('<p class="fine"><b>Low certainty:</b> no class reached 50%, so the model is '
                       'unsure about this text.</p>')
            explain = (f'<p class="fine" style="margin-top:1rem">The model chose <b>{name}</b> because these '
                       f'words or phrases in your text carry the strongest learned weight toward that class:</p>'
                       f'<div class="chips">{chips}</div>') if chips else (
                       '<p class="fine" style="margin-top:1rem">None of the words in this text were '
                       'strongly linked to one class in the training data.</p>')
            st.markdown(f"""
<div class="result" style="--tone:{color}">
  <div class="result-head"><span class="result-label">{name}</span>{conf_html}</div>
  <div style="margin-top:1rem">{bars}</div>
  {low}{explain}
  <p class="fine" style="margin-top:.9rem">Probabilities are the model's own estimates, not a guarantee
  of being right. The word list shows learned statistical weights; it is not a human-style reason.</p>
</div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ C: dataset
with tab_data:
    st.subheader("Dataset overview")
    if not artifacts:
        st.info("Train a model to see the statistics of the dataset it was trained on.")
    else:
        d = artifacts["metrics"]["dataset"]
        counts = d["class_counts"]
        if d["is_sample"]:
            st.warning("The loaded model was trained on the small bundled DEMO dataset. "
                       "It only shows that the application works; it cannot support claims about "
                       "real-world accuracy. See the README to train on a proper dataset.")
        stat_cards([(f'{d["total"]:,}', "Total records (after cleaning)"),
                    (f'{counts.get("positive", 0):,}', "Positive", TONES["positive"][0]),
                    (f'{counts.get("negative", 0):,}', "Negative", TONES["negative"][0]),
                    (f'{counts.get("neutral", 0):,}', "Neutral", TONES["neutral"][0])])
        c1, c2 = st.columns([1.1, 1], gap="large")
        with c1:
            fig = distribution_figure(counts)
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            cl = d["cleaning"]
            st.markdown(f"**Source:** `{d['name']}`  \n"
                        f"**Columns used:** text = `{cl['text_column']}`, label = `{cl['label_column']}`")
            clean_table = pd.DataFrame({
                "Step": ["Rows read", "Missing text removed", "Missing labels removed",
                         "Unsupported labels removed", "Empty after cleaning removed",
                         "Duplicates removed", "Rows kept"],
                "Count": [cl["rows_read"], cl["missing_text"], cl["missing_label"],
                          cl["invalid_label"], cl["empty_after_cleaning"],
                          cl["duplicates_removed"], cl["rows_kept"]]})
            st.dataframe(clean_table, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Train on a dataset")
    st.caption("Upload a labelled CSV with a text column and a sentiment column "
               "(positive / negative / neutral, case does not matter). "
               "Nothing is trained until you press the button.")
    upload = st.file_uploader("Labelled CSV file", type=["csv"], label_visibility="collapsed")
    if upload is not None:
        data = upload.getvalue()
        try:
            columns = list(read_csv_safely(io.BytesIO(data)).columns)
            guess_text, guess_label = detect_columns(columns)
            u1, u2 = st.columns(2)
            text_col = u1.selectbox("Text column", columns,
                                    index=columns.index(guess_text) if guess_text in columns else 0)
            label_col = u2.selectbox("Sentiment column", columns,
                                     index=columns.index(guess_label) if guess_label in columns else min(1, len(columns) - 1))
            df, rep = load_dataset(io.BytesIO(data), text_col, label_col)
            st.success(f"The file is valid: {rep['rows_kept']:,} usable rows out of {rep['rows_read']:,}.")
            stat_cards([(f"{rep['rows_kept']:,}", "Usable records"),
                        (f"{rep['class_counts']['positive']:,}", "Positive", TONES["positive"][0]),
                        (f"{rep['class_counts']['negative']:,}", "Negative", TONES["negative"][0]),
                        (f"{rep['class_counts']['neutral']:,}", "Neutral", TONES["neutral"][0])])
            dropped = rep["rows_read"] - rep["rows_kept"]
            if dropped:
                st.caption(f"{dropped:,} rows were dropped: {rep['missing_text']} missing text, "
                           f"{rep['missing_label']} missing label, {rep['invalid_label']} unsupported label, "
                           f"{rep['empty_after_cleaning']} empty after cleaning, "
                           f"{rep['duplicates_removed']} duplicates.")
            if rep["invalid_label"]:
                st.warning(f"Unsupported labels such as {rep['invalid_label_examples']} were ignored.")
            st.dataframe(df[["text", "sentiment"]].head(8), hide_index=True, width="stretch")
            if st.button("Train model on this dataset", type="primary"):
                train_and_report(io.BytesIO(data), text_col=text_col, label_col=label_col,
                                 source_name=upload.name)
        except DatasetError as exc:
            st.error(f"This file cannot be used: {exc}")
        except Exception as exc:
            st.error(f"The file could not be processed: {exc}")

# ------------------------------------------------------------------ D: performance
with tab_perf:
    st.subheader("Model performance")
    if not artifacts:
        st.info("Train a model to see its evaluation results.")
    else:
        m = artifacts["metrics"]
        t, sp = m["test"], m["split"]
        st.caption(f"All numbers below come from the TEST set: {sp['n_test']:,} texts the model never saw "
                   f"during training (the other {sp['n_train']:,} were used for training). "
                   f"Split was {'stratified' if sp['stratified'] else 'random'}, seed {m['random_seed']}.")
        for w in m["warnings"]:
            st.warning(w)
        stat_cards([(pct(t["accuracy"]), "Accuracy"),
                    (pct(t["precision_macro"]), "Precision (macro)"),
                    (pct(t["recall_macro"]), "Recall (macro)"),
                    (pct(t["f1_macro"]), "F1-score (macro)")])

        fig = tm.make_evaluation_figure(m)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("##### Classification report (test set)")
        rep_df = pd.DataFrame(t["report"]).T.rename(columns={"f1-score": "f1-score"})
        rep_df["support"] = rep_df["support"].astype(int)
        st.dataframe(rep_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}),
                     width="stretch")

        st.markdown("##### Model comparison")
        comp = pd.DataFrame(m["comparison"]).T
        comp.columns = ["Cross-validated macro F1 (train only)", "Test accuracy", "Test macro F1"]
        comp["Chosen"] = ["yes" if n == m["model_name"] else "" for n in comp.index]
        st.dataframe(comp.style.format({c: "{:.3f}" for c in comp.columns[:3]}, na_rep="n/a"),
                     width="stretch")
        st.caption(f"Selection rule: {m['selection_rule']}")
        st.markdown('<p class="fine">A high score on a small or unrepresentative dataset does not guarantee '
                    'the same accuracy on real-world text.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ E: about
with tab_about:
    st.subheader("About the project")
    topics = [
        ("Natural Language Processing (NLP)",
         "NLP is the field of computer science that teaches computers to work with human language. "
         "Computers only understand numbers, so the first challenge is turning words into numbers "
         "in a way that keeps their meaning useful."),
        ("Text preprocessing",
         "Before learning, every text is tidied: lower-cased, links and @mentions replaced with "
         "placeholders, extra symbols and spaces removed. Negations such as <i>not</i>, <i>never</i> and "
         "<i>isn't</i> are deliberately kept, because 'not good' means the opposite of 'good'."),
        ("TF-IDF feature extraction",
         "TF-IDF gives each word or two-word phrase a score. Words that appear often in a text get a "
         "higher score (term frequency), but words that appear in almost every text, like 'the', are "
         "scaled down (inverse document frequency). Each text becomes a list of scores the model can use."),
        ("Logistic Regression",
         "A classifier that learns a weight for every word and phrase. For a new text it adds up the "
         "weights, turns the totals into probabilities for positive, negative and neutral, and picks "
         "the most likely class. Because the weights are readable, we can show which words mattered."),
        ("Model evaluation",
         "The data is split into a training set and a test set. The model learns only from the training "
         "set and is graded on the test set. <b>Accuracy</b> is the share of correct predictions; "
         "<b>precision</b> asks 'when it says positive, how often is it right?'; <b>recall</b> asks 'of all "
         "truly positive texts, how many did it find?'; <b>F1</b> balances the two. The confusion matrix "
         "shows exactly which classes get mixed up."),
    ]
    for title, body in topics:
        st.markdown(f'<div class="card" style="margin-bottom:.9rem"><h4>{title}</h4><p>{body}</p></div>',
                    unsafe_allow_html=True)
