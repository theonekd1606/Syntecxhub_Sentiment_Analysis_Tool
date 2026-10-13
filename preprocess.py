"""
preprocess.py - text cleaning and dataset loading for the Sentiment AI project.

Two jobs live here:
  1. clean_text()    -> turns raw text into a tidy form the model can learn from.
  2. load_dataset()  -> reads a CSV, validates it, normalises the labels and
                        returns a clean DataFrame plus a report of what was fixed.

The SAME clean_text() function is used for training and for prediction, so the
model always sees text prepared in exactly the same way.
"""

import re

import pandas as pd

# The three classes this project predicts, in the order we always display them.
LABELS = ["negative", "neutral", "positive"]

# Every spelling we accept for a label, mapped to one of the three classes.
# Numeric labels follow the two most common conventions:
#   0/1       -> negative/positive  (binary datasets)
#   0/2/4     -> negative/neutral/positive  (Sentiment140 style)
# If your numbers mean something else, rename them in the CSV first.
LABEL_MAP = {
    "positive": "positive", "pos": "positive", "p": "positive", "good": "positive",
    "1": "positive", "4": "positive", "+1": "positive",
    "negative": "negative", "neg": "negative", "n": "negative", "bad": "negative",
    "0": "negative", "-1": "negative",
    "neutral": "neutral", "neu": "neutral", "mixed": "neutral", "2": "neutral",
}

# Column names we look for automatically (compared in lower case).
TEXT_COLUMNS = ["text", "review", "reviews", "tweet", "comment", "content",
                "sentence", "feedback", "message", "review_text", "reviewtext", "body"]
LABEL_COLUMNS = ["sentiment", "airline_sentiment", "label", "labels", "target", "polarity", "class", "category"]

MAX_TEXT_LENGTH = 5000  # longest input the app accepts, in characters


class DatasetError(ValueError):
    """Raised when a dataset cannot be used. The message is safe to show to the user."""


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_MENTION = re.compile(r"@\w+")
_HTML_TAG = re.compile(r"<[^>]+>")
_POSSESSIVE = re.compile(r"'s\b")
_REPEATED_LETTERS = re.compile(r"([a-z])\1{2,}")      # "sooooo" -> "soo"
_REPEATED_PUNCT = re.compile(r"([!?])\1+")            # "!!!" -> "!"
_NOT_ALLOWED = re.compile(r"[^a-z0-9!?\s]")
_SPACES = re.compile(r"\s+")

# Contractions are expanded so negations survive as the separate word "not".
# "isn't" -> "is not", "don't" -> "do not", "can't" -> "can not"
_CONTRACTIONS = [
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"), " would"),
]


def clean_text(text) -> str:
    """Clean one piece of text. Never raises; returns "" for missing/invalid input.

    Steps (stop words are deliberately NOT removed, because words like
    "not", "no" and "never" carry sentiment):
      1. lower-case, 2. replace URLs and @mentions with placeholders,
      3. strip HTML tags, 4. expand contractions (keep negations),
      5. keep letters, digits, "!" and "?" only, 6. tidy repeated letters/spaces.
    """
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""
    text = str(text).lower().replace("\u2019", "'").replace("\u2018", "'")
    text = _URL.sub(" url ", text)
    text = _MENTION.sub(" user ", text)
    text = _HTML_TAG.sub(" ", text)
    text = text.replace("#", " ")                      # keep hashtag words, drop the symbol
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    text = _POSSESSIVE.sub("", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    text = _NOT_ALLOWED.sub(" ", text)
    text = _REPEATED_LETTERS.sub(r"\1\1", text)
    text = re.sub(r"([!?])", r" \1 ", text)           # make "!" and "?" their own tokens
    return _SPACES.sub(" ", text).strip()


# --------------------------------------------------------------------------
# Labels and columns
# --------------------------------------------------------------------------
def normalize_label(value):
    """Return 'positive', 'negative' or 'neutral', or None if the label is unsupported."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    key = str(value).strip().lower()
    try:                                   # turn "4.0" into "4"
        number = float(key)
        if number == int(number):
            key = str(int(number))
    except ValueError:
        pass
    return LABEL_MAP.get(key)


def detect_columns(columns):
    """Guess (text_column, label_column) from a list of column names. Either may be None."""
    lookup = {str(c).strip().lower(): c for c in columns}
    text_col = next((lookup[name] for name in TEXT_COLUMNS if name in lookup), None)
    label_col = next((lookup[name] for name in LABEL_COLUMNS if name in lookup), None)
    return text_col, label_col


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------
def read_csv_safely(source) -> pd.DataFrame:
    """Read a CSV from a path or file-like object, trying common encodings."""
    last_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except FileNotFoundError:
            raise DatasetError(f"Dataset file not found: {source}")
        except pd.errors.EmptyDataError:
            raise DatasetError("The CSV file is empty.")
        except pd.errors.ParserError as exc:
            raise DatasetError(f"The file could not be read as a CSV: {exc}")
    raise DatasetError(f"The file encoding is not supported: {last_error}")


def load_dataset(source, text_col=None, label_col=None):
    """Load, validate and clean a labelled sentiment CSV.

    Returns (df, report):
      df     has columns text, clean_text, sentiment
      report is a dict describing every row that was dropped and why.
    Raises DatasetError with a friendly message if the data cannot be used.
    """
    raw = read_csv_safely(source)
    if raw.empty:
        raise DatasetError("The CSV has a header but no rows.")

    guessed_text, guessed_label = detect_columns(raw.columns)
    text_col = text_col or guessed_text
    label_col = label_col or guessed_label
    for role, col in (("text", text_col), ("sentiment label", label_col)):
        if col is None:
            raise DatasetError(
                f"Could not find the {role} column. Columns in your file: "
                f"{', '.join(map(str, raw.columns))}. Rename them to 'text' and 'sentiment' "
                f"or choose them explicitly.")
        if col not in raw.columns:
            raise DatasetError(f"Column '{col}' does not exist. Columns found: "
                               f"{', '.join(map(str, raw.columns))}.")
    if text_col == label_col:
        raise DatasetError("The text column and the label column must be different.")

    df = raw[[text_col, label_col]].rename(columns={text_col: "text", label_col: "sentiment_raw"})
    report = {"rows_read": int(len(df)), "text_column": str(text_col), "label_column": str(label_col)}

    # 1. missing text / missing labels
    missing_text = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
    report["missing_text"] = int(missing_text.sum())
    df = df[~missing_text]
    missing_label = df["sentiment_raw"].isna()
    report["missing_label"] = int(missing_label.sum())
    df = df[~missing_label]

    # 2. label normalisation (handles upper/lower case and spelling variants)
    df = df.assign(sentiment=df["sentiment_raw"].map(normalize_label))
    invalid = df["sentiment"].isna()
    report["invalid_label"] = int(invalid.sum())
    report["invalid_label_examples"] = sorted(map(str, df.loc[invalid, "sentiment_raw"].unique()))[:5]
    df = df[~invalid]

    # 3. clean the text, drop rows with nothing left
    df = df.assign(text=df["text"].astype(str), clean_text=df["text"].map(clean_text))
    empty = df["clean_text"] == ""
    report["empty_after_cleaning"] = int(empty.sum())
    df = df[~empty]

    # 4. duplicates (same cleaned text). Keeping them could put the same sentence
    #    in both the training and the test set, which would inflate the scores.
    before = len(df)
    df = df.drop_duplicates(subset="clean_text", keep="first")
    report["duplicates_removed"] = int(before - len(df))

    df = df[["text", "clean_text", "sentiment"]].reset_index(drop=True)
    report["rows_kept"] = int(len(df))
    report["class_counts"] = {label: int((df["sentiment"] == label).sum()) for label in LABELS}

    if df.empty:
        hint = ""
        if report["invalid_label"]:
            hint = (f" Unsupported labels found, e.g. {report['invalid_label_examples']}. "
                    f"Supported: positive, negative, neutral (or pos/neg/neu, 0/1, 0/2/4).")
        raise DatasetError("No usable rows remain after cleaning." + hint)
    return df, report
