"""
train_model.py - trains, evaluates, saves and loads the sentiment model.

Run it from the command line:
    python train_model.py                       # uses data/sentiment_dataset.csv
    python train_model.py --data path/to.csv    # use any labelled CSV

What happens when you run it:
  1. The CSV is loaded, validated and cleaned (preprocess.py).
  2. The data is split into a TRAINING set and a TEST set (stratified if possible).
  3. Logistic Regression and Multinomial Naive Bayes are compared using
     cross-validation on the TRAINING set only. The test set is never used to choose.
  4. The TF-IDF vectoriser is fitted on the training texts only (no data leakage).
  5. Both models are scored on the untouched TEST set; the chosen one is saved.
  6. Model, vectoriser, metrics (JSON) and an evaluation chart are written to disk.

This file also contains load_artifacts() and predict_text(), which the Streamlit
app uses, so training and prediction share one source of truth.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts to files/memory; no pop-up window needed
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocess import LABELS, MAX_TEXT_LENGTH, DatasetError, clean_text, load_dataset

# ---------------------------------------------------------------- paths & settings
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

DEFAULT_DATASET = DATA_DIR / "sentiment_dataset.csv"          # your real dataset
SAMPLE_DATASET = DATA_DIR / "sample_sentiment_dataset.csv"    # tiny demo data only
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
REPORT_PATH = REPORT_DIR / "model_evaluation.png"

RANDOM_SEED = 42
TEST_SIZE = 0.2
MIN_ROWS = 20  # fewer usable rows than this cannot give a meaningful split

SELECTION_RULE = ("Highest mean 5-fold cross-validated macro F1 on the training set only. "
                  "Logistic Regression is the primary model and is kept unless Naive Bayes "
                  "beats it by more than 0.01.")

# Chart colours (match the app theme)
ORANGE, PANEL, LINE, TEXT, MUTED = "#ff7a1a", "#141414", "#2a2a2a", "#f5f5f5", "#9a9a9a"


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model files are missing."""


# ---------------------------------------------------------------- building blocks
def build_vectorizer(n_documents: int) -> TfidfVectorizer:
    """TF-IDF with single words + word pairs. min_df=1 for small data, 2 for big data."""
    return TfidfVectorizer(
        ngram_range=(1, 2),        # "good" and "not good"
        max_features=20000,
        min_df=1 if n_documents < 2000 else 2,
        sublinear_tf=True,         # dampen very frequent words
        token_pattern=r"(?u)\b\w+\b|[!?]",   # keep one-letter words and ! ?
    )


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            C=5.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
    }


def split_data(df, test_size=TEST_SIZE, seed=RANDOM_SEED):
    """Train/test split. Stratified (same class mix in both) whenever the data allows it."""
    y = df["sentiment"]
    n_classes = y.nunique()
    n_test = int(round(len(df) * test_size))
    can_stratify = (y.value_counts().min() >= 2 and n_test >= n_classes
                    and len(df) - n_test >= n_classes)
    warnings = []
    if not can_stratify:
        warnings.append("Some class has too few examples for a stratified split, so a plain "
                        "random split was used. Class proportions in train/test may differ.")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], y, test_size=test_size, random_state=seed,
        stratify=y if can_stratify else None)
    return X_train, X_test, y_train, y_test, can_stratify, warnings


def cross_validate_models(models, X_train, y_train, seed=RANDOM_SEED):
    """Mean macro-F1 per model using only the training data. Returns (scores, folds)."""
    folds = min(5, int(y_train.value_counts().min()))
    if folds < 2:
        return None, folds
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = {}
    for name, model in models.items():
        # The vectoriser sits INSIDE the pipeline so it is re-fitted on each fold's
        # training part only - validation folds never leak into it.
        pipe = Pipeline([("tfidf", build_vectorizer(len(X_train))), ("clf", clone(model))])
        scores[name] = float(cross_val_score(pipe, X_train, y_train, cv=cv,
                                             scoring="f1_macro").mean())
    return scores, folds


def choose_model(cv_scores):
    if not cv_scores:
        return "Logistic Regression"
    lr, nb = cv_scores["Logistic Regression"], cv_scores["Multinomial Naive Bayes"]
    return "Multinomial Naive Bayes" if nb > lr + 0.01 else "Logistic Regression"


def evaluate(model, X, y, labels):
    """All metrics computed from real predictions on the given (test) data."""
    pred = model.predict(X)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, average="macro",
                                                 zero_division=0)
    return {
        "accuracy": float((pred == y.to_numpy()).mean()),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "report": classification_report(y, pred, labels=labels, target_names=labels,
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


# ---------------------------------------------------------------- charts
def _style(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.tick_params(colors=MUTED)
    ax.title.set_color(TEXT)


def make_evaluation_figure(metrics: dict):
    """Confusion matrix + per-class scores, built from the saved real test results."""
    labels = metrics["labels"]
    cm = np.array(metrics["test"]["confusion_matrix"])
    report = metrics["test"]["report"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=PANEL)
    cmap = LinearSegmentedColormap.from_list("orange_dark", [PANEL, ORANGE])
    ax1.imshow(cm, cmap=cmap)
    ax1.set_xticks(range(len(labels)), labels)
    ax1.set_yticks(range(len(labels)), labels)
    ax1.set_xlabel("Predicted", color=MUTED)
    ax1.set_ylabel("Actual", color=MUTED)
    ax1.set_title("Confusion matrix (test set)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            dark_cell = cm[i, j] > 0.6 * cm.max() if cm.max() else False
            ax1.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14,
                     fontweight="bold", color="#0a0a0a" if dark_cell else TEXT)
    _style(ax1)

    x = np.arange(len(labels))
    width = 0.26
    for k, (key, shade) in enumerate([("precision", "#ffb27a"), ("recall", ORANGE),
                                      ("f1-score", "#c24f00")]):
        ax2.bar(x + (k - 1) * width, [report[l][key] for l in labels], width,
                label=key, color=shade)
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Per-class scores (test set)")
    ax2.grid(axis="y", color=LINE, linewidth=0.6)
    ax2.set_axisbelow(True)
    legend = ax2.legend(facecolor=PANEL, edgecolor=LINE)
    for t in legend.get_texts():
        t.set_color(TEXT)
    _style(ax2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- training
def train(source=None, text_col=None, label_col=None, source_name=None, is_sample=False,
          test_size=TEST_SIZE, seed=RANDOM_SEED, log=print):
    """Train, evaluate and save. Returns the metrics dict. Raises DatasetError if unusable."""
    source = source if source is not None else (DEFAULT_DATASET if DEFAULT_DATASET.exists()
                                                else SAMPLE_DATASET)
    if isinstance(source, (str, Path)):
        source = Path(source)
        source_name = source_name or source.name
        is_sample = is_sample or source.resolve() == SAMPLE_DATASET.resolve()
    source_name = source_name or "uploaded file"

    log(f"Loading dataset: {source_name}")
    df, report = load_dataset(source, text_col, label_col)
    labels = [l for l in LABELS if (df["sentiment"] == l).any()]
    if len(df) < MIN_ROWS:
        raise DatasetError(f"Only {len(df)} usable rows - at least {MIN_ROWS} are needed to train "
                           f"and test a model.")
    if len(labels) < 2:
        raise DatasetError(f"Only one sentiment class found ({labels[0]}). "
                           f"At least two different classes are needed.")

    warnings = []
    if is_sample:
        warnings.append("This is the small demo dataset. Scores on it are NOT a reliable "
                        "estimate of real-world accuracy.")
    if len(labels) < 3:
        missing = [l for l in LABELS if l not in labels]
        warnings.append(f"No examples of: {', '.join(missing)}. The model cannot predict those classes.")
    if len(df) < 1000 and not is_sample:
        warnings.append("The dataset has fewer than 1,000 rows; treat the scores with caution.")

    X_train, X_test, y_train, y_test, stratified, split_warnings = split_data(df, test_size, seed)
    warnings += split_warnings
    log(f"Split: {len(X_train)} training rows, {len(X_test)} test rows "
        f"({'stratified' if stratified else 'random'})")

    models = build_models()
    cv_scores, folds = cross_validate_models(models, X_train, y_train, seed)
    if cv_scores is None:
        warnings.append("Too few examples per class for cross-validation; "
                        "Logistic Regression was used by default.")
    selected = choose_model(cv_scores)

    # Fit the vectoriser on TRAINING texts only, then just transform the test texts.
    vectorizer = build_vectorizer(len(X_train))
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    comparison, fitted = {}, {}
    for name, model in models.items():
        model.fit(Xtr, y_train)
        fitted[name] = model
        result = evaluate(model, Xte, y_test, labels)
        comparison[name] = {"cv_f1_macro": (cv_scores or {}).get(name),
                            "test_accuracy": result["accuracy"],
                            "test_f1_macro": result["f1_macro"]}
        log(f"  {name}: test accuracy {result['accuracy']:.3f}, macro F1 {result['f1_macro']:.3f}")
        if name == selected:
            selected_result = result
    log(f"Selected model: {selected}")

    metrics = {
        "model_name": selected,
        "selection_rule": SELECTION_RULE,
        "labels": labels,
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "random_seed": seed,
        "dataset": {"name": source_name, "is_sample": bool(is_sample), "cleaning": report,
                    "class_counts": report["class_counts"], "total": report["rows_kept"]},
        "split": {"n_train": int(len(X_train)), "n_test": int(len(X_test)),
                  "test_size": test_size, "stratified": bool(stratified), "cv_folds": folds},
        "comparison": comparison,
        "test": selected_result,
        "warnings": warnings,
    }

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[selected], MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig = make_evaluation_figure(metrics)
    fig.savefig(REPORT_PATH, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    log(f"Saved: {MODEL_PATH.name}, {VECTORIZER_PATH.name}, {METRICS_PATH.name}, {REPORT_PATH.name}")
    return metrics


# ---------------------------------------------------------------- prediction
def artifacts_exist() -> bool:
    return all(p.exists() for p in (MODEL_PATH, VECTORIZER_PATH, METRICS_PATH))


def load_artifacts() -> dict:
    """Load the saved model, vectoriser and metrics. Raises ModelNotFoundError if missing."""
    missing = [p.name for p in (MODEL_PATH, VECTORIZER_PATH, METRICS_PATH) if not p.exists()]
    if missing:
        raise ModelNotFoundError(f"Missing model files: {', '.join(missing)}. "
                                 f"Train the model first with: python train_model.py")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    note = None
    if metrics.get("sklearn_version") != sklearn.__version__:
        note = (f"The model was trained with scikit-learn {metrics.get('sklearn_version')} but "
                f"{sklearn.__version__} is installed. If predictions fail, retrain the model.")
    return {"model": joblib.load(MODEL_PATH), "vectorizer": joblib.load(VECTORIZER_PATH),
            "metrics": metrics, "version_note": note}


def predict_text(text, artifacts: dict, top_terms: int = 6) -> dict:
    """Predict sentiment for one text with the saved model. Raises ValueError on bad input."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter some text to analyse.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"The text is too long ({len(text):,} characters). "
                         f"The limit is {MAX_TEXT_LENGTH:,}.")
    cleaned = clean_text(text)
    real_words = [w for w in cleaned.split() if w.isalpha() and w not in ("url", "user")]
    if not real_words:
        raise ValueError("The text has no readable words (only symbols, numbers, links or "
                         "mentions). Please write a sentence.")

    model, vectorizer = artifacts["model"], artifacts["vectorizer"]
    features = vectorizer.transform([cleaned])
    label = str(model.predict(features)[0])
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probabilities = {str(c): float(p) for c, p in zip(model.classes_, model.predict_proba(features)[0])}

    return {"label": label, "probabilities": probabilities,
            "confidence": probabilities.get(label), "cleaned_text": cleaned,
            "top_terms": _influential_terms(model, vectorizer, features, label, top_terms)}


def _influential_terms(model, vectorizer, features, label, n):
    """Words/phrases in the text that push the model toward the predicted class.

    For Logistic Regression: tf-idf value x the learned weight for that class.
    For Naive Bayes: tf-idf value x (log-probability for this class minus the average
    of the other classes). This describes the model's learned weights; it is not proof
    of how a person would read the sentence.
    """
    classes = list(model.classes_)
    idx = classes.index(label)
    if hasattr(model, "coef_"):
        coef = model.coef_
        weights = coef[idx] if coef.shape[0] > 1 else (coef[0] if idx == 1 else -coef[0])
    elif hasattr(model, "feature_log_prob_"):
        flp = model.feature_log_prob_
        others = [i for i in range(len(classes)) if i != idx]
        weights = flp[idx] - flp[others].mean(axis=0)
    else:
        return []
    row = features.tocoo()
    names = vectorizer.get_feature_names_out()
    scored = sorted(((float(v * weights[j]), names[j]) for j, v in zip(row.col, row.data)),
                    reverse=True)
    return [term for score, term in scored[:n] if score > 0]


# ---------------------------------------------------------------- command line
def main():
    parser = argparse.ArgumentParser(description="Train the Sentiment AI model.")
    parser.add_argument("--data", help="Path to a labelled CSV (default: data/sentiment_dataset.csv, "
                                       "or the demo sample if that file does not exist).")
    parser.add_argument("--text-column", help="Name of the text column (auto-detected if omitted).")
    parser.add_argument("--label-column", help="Name of the sentiment column (auto-detected if omitted).")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                        help="Fraction of data kept for testing (default 0.2).")
    args = parser.parse_args()
    if not 0.05 <= args.test_size <= 0.5:
        sys.exit("--test-size must be between 0.05 and 0.5")
    try:
        if args.data is None and not DEFAULT_DATASET.exists():
            print("NOTE: data/sentiment_dataset.csv not found - training on the small DEMO sample.\n"
                  "      Its scores are not reliable. See README.md to add a real dataset.\n")
        metrics = train(args.data, args.text_column, args.label_column, test_size=args.test_size)
    except DatasetError as exc:
        sys.exit(f"Dataset problem: {exc}")
    test = metrics["test"]
    print(f"\nTest accuracy: {test['accuracy']:.3f} | macro precision: {test['precision_macro']:.3f} | "
          f"macro recall: {test['recall_macro']:.3f} | macro F1: {test['f1_macro']:.3f}")
    for w in metrics["warnings"]:
        print("WARNING:", w)
    print("\nNext step: streamlit run app.py")


if __name__ == "__main__":
    main()
