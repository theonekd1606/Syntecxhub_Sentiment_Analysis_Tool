# SENTIMENT AI - Sentiment Analysis Tool using Machine Learning

> Understand the emotion behind every word.

Sentiment AI reads a piece of text (a tweet, a product review, a movie review, user feedback) and predicts whether its sentiment is **positive**, **negative** or **neutral**. The predictions come from a real machine-learning model (TF-IDF + Logistic Regression) trained on labelled text. There are no keyword rules and no hard-coded answers.

**Supported Python:** 3.10, 3.11 or 3.12 (3.11 recommended). Developed and tested on Python 3.12.

## Features

- Streamlit dashboard in an orange-and-black theme: text analysis, dataset overview, model performance, and an "About" page for presenting.
- Prediction with class probabilities, plus the words/phrases that influenced the model most.
- Clean-up of messy CSVs: missing text, missing labels, duplicates, inconsistent label capitalisation, unsupported labels.
- Leak-free training: the TF-IDF vectoriser and the classifier only ever see the training split.
- Logistic Regression (primary) compared with Multinomial Naive Bayes; the winner is chosen by a stated rule, using cross-validation on training data only.
- Every metric shown in the app is computed from the real test set. Nothing is invented.
- Model saved with Joblib and reloaded on start-up; the app never retrains by itself.

## Technologies

Python, Pandas, NumPy, regular expressions for preprocessing (no NLTK downloads needed), Scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, metrics), Matplotlib, Streamlit, Joblib.

## Project structure

```
Sentiment_Analysis_Tool/
├── app.py                  Streamlit interface
├── train_model.py          training, evaluation, saving, loading, prediction
├── preprocess.py           text cleaning + dataset loading/validation
├── requirements.txt
├── README.md
├── .streamlit/config.toml  dark orange theme settings
├── data/
│   ├── sample_sentiment_dataset.csv   255-row DEMO data (not for real evaluation)
│   └── sentiment_dataset.csv          <- YOUR dataset goes here (you add this file)
├── models/                 created by training
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── metrics.json        real test results shown in the app
└── reports/
    └── model_evaluation.png   confusion matrix + per-class scores
```

Two small deviations from the suggested layout: the vectoriser and metrics are saved as separate files (the app needs them), and the demo data has its own file name so it can never be mistaken for a real dataset.

## Installation (Windows, PowerShell)

1. Install Python 3.11 from python.org (tick "Add python.exe to PATH").
2. Open PowerShell inside the project folder (in File Explorer: open the folder, click the address bar, type `powershell`, press Enter).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell says "running scripts is disabled", run this once and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If `py -3.11` is not found, use `python -m venv .venv` instead.

## Preparing the dataset

The app needs a CSV with a **text column** and a **sentiment column**.

- Recognised text columns: `text`, `review`, `tweet`, `comment`, `content`, `sentence`, `feedback`, `message`, and similar.
- Recognised label columns: `sentiment`, `airline_sentiment`, `label`, `target`, `polarity`, `class`, and similar.
- Recognised labels (any capitalisation): `positive/pos`, `negative/neg`, `neutral/neu`. Numbers: `0` = negative and `1` = positive (binary), or `0/2/4` = negative/neutral/positive (Sentiment140 style). Rows with other labels are skipped and reported.
- Different column names? Pass `--text-column` and `--label-column`, or pick the columns in the app.

**Using a real public dataset.** Download one yourself from its official page (the project never downloads files automatically), save it as `data\sentiment_dataset.csv`, and make sure it has a header row. Well-known examples:

| Dataset | Classes | Note |
|---|---|---|
| Twitter US Airline Sentiment (Kaggle) | positive / negative / neutral | Best fit: it has all three classes. The label column is usually `airline_sentiment`. |
| IMDB Movie Reviews (50k) | positive / negative | No neutral class, so the model will not predict neutral. |
| Sentiment140 | negative / positive (0/4) | Usually has no header row; add one (`target,id,date,flag,user,text`) before use. |

Check each dataset's licence and terms before using it in a published project.

**No dataset yet?** Training falls back to `data\sample_sentiment_dataset.csv`, a small hand-written set of 255 sentences. It only proves the application works. Its scores are *not* reliable, and the app labels the model "Demo dataset".

## Training the model

```powershell
python train_model.py
python train_model.py --data data\my_file.csv --text-column review --label-column stars_label
```

This prints the split sizes, compares the two models, saves the files in `models\`, and writes `reports\model_evaluation.png`. You can also upload a CSV and train from the **Dataset** tab of the app.

## Launching the app

```powershell
streamlit run app.py
```

Your browser opens at http://localhost:8501. Press `Ctrl+C` in PowerShell to stop it.

## Example predictions

These come from the saved model, so they can change when you train on a different dataset. With the bundled demo data they were:

| Text | Prediction |
|---|---|
| This product is absolutely amazing! | Positive |
| I had a terrible experience with this service. | Negative |
| The package arrived this morning. | Neutral |
| I thought it would be bad, but I actually loved it. | Positive |
| I do not recommend this product. | Negative |

A model trained on 255 sentences is only roughly right (see Limitations). The probabilities in the app show how sure it is.

## Understanding the metrics

All metrics are calculated on the **test set**: texts the model never saw while learning.

- **Accuracy** - share of all predictions that were correct. It can mislead when one class is much bigger than the others.
- **Precision** - when the model says "positive", how often is it right?
- **Recall** - of all the truly positive texts, how many did the model find?
- **F1-score** - one number balancing precision and recall.
- **Macro average** - the average over the classes, counting each class equally.
- **Confusion matrix** - rows are the true class, columns the predicted class. The diagonal holds correct predictions; everything else is a mistake, and its position shows which classes get confused.

## How the model is chosen

1. 80% of the data is used for training and 20% is held back for testing (stratified when possible, seed 42).
2. Logistic Regression and Naive Bayes are compared with 5-fold cross-validation on the **training data only**, using macro F1.
3. Logistic Regression is the primary model. It is replaced only if Naive Bayes wins by more than 0.01.
4. The chosen model is then scored on the test set.

## Limitations and future improvements

- A bag-of-words model does not truly understand sarcasm, context or long-range meaning. Word pairs help with cases like "not good", but subtle cases will fail.
- It only knows the vocabulary and domain of its training data. A model trained on movie reviews will do worse on tweets.
- The demo dataset is tiny; scores on it say nothing about real-world accuracy.
- Probabilities are the model's own estimates, not guarantees, and can be over- or under-confident.
- Only English text is supported.

Ideas for later: train on a larger dataset, tune hyperparameters with grid search, add character n-grams, try a linear SVM, compare with a pre-trained transformer, add batch CSV prediction.

## Troubleshooting

- *"No trained model found"* - run `python train_model.py` first.
- *scikit-learn version warning, or the model fails to load* - retrain with `python train_model.py`. Saved models are tied to the scikit-learn version that created them.
- *Dataset errors* - the message names the problem (missing column, unsupported labels, too few rows). Fix the CSV or choose the right columns.
