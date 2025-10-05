# Sentiment Analysis on IMDB Reviews

This repository contains an end-to-end experiment for binary sentiment classification of IMDB movie reviews. The main artifact is a Jupyter notebook that trains an LSTM-based model and evaluates its performance on the IMDB reviews dataset.

Key points:
- Dataset: `data/IMDB Dataset.csv` (CSV with review text and sentiment label).
- Notebook: `notebooks/IMDB_reviews_Sentiment_analysis_LSTM.ipynb` — contains data loading, preprocessing, model building (LSTM), training, and evaluation.
- Models/artifacts: Save trained weights or exported models to the `models/` directory.

## Repository structure

- `data/` — dataset files (the project expects `IMDB Dataset.csv` here).
- `notebooks/` — Google Colab notebook(s) with preprocessing, model training and evaluation.`IMDB_reviews_Sentiment_analysis_LSTM.ipynb`.
- `models/` — trained models.
- `src/` — project source code.
- `README.md`

## About the project

The goal is to build a robust binary classifier that labels IMDB reviews as positive or negative. The notebook demonstrates a typical NLP model workflow:

1. Exploratory data analysis (class balance, sample lengths, etc.).
2. Text preprocessing (tokenization, lowercasing, optional stopword removal, padding/truncation).
3. Converting text to sequences using an embedding layer (trainable or pretrained embeddings).
4. Building an LSTM-based neural network to model sequential text data.
5. Training with validation, checkpointing, and simple logging.
6. Evaluation using accuracy, precision, recall, F1-score, and a confusion matrix.

## Quick start — recommended steps

1. Create a Python environment (recommended Python 3.8+):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install the common dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk tensorflow tqdm
```

3. Open and run the notebook:

```bash
jupyter lab notebooks/IMDB_reviews_Sentiment_analysis_LSTM.ipynb
```

The notebook is organized so you can run cells top-to-bottom. It includes data loading, preprocessing, model definition, training, and evaluation.

# Sentiment Analysis on IMDB Reviews (Google Colab first)

This repository contains an end-to-end experiment for binary sentiment classification of IMDB movie reviews. The primary workflow is provided as a notebook that's convenient to run on Google Colab (recommended). A local fallback is included below.

Key points:
- Dataset: `data/IMDB Dataset.csv` (CSV with review text and sentiment label).
- Notebook: `notebooks/IMDB_reviews_Sentiment_analysis_LSTM.ipynb` — designed to run on Colab (includes cells for Drive mounting, deps, training, evaluation).
- Models: Save trained weights or exported models to the `models/` directory (recommended to persist to Drive when using Colab).

## Colab-first quick start

Preferred: open the notebook directly in Google Colab. There are two easy ways:

- Open from this repository on GitHub using the Colab URL:

	https://colab.research.google.com/github/Ruran8wa/Sentiment_analysis/blob/main/notebooks/IMDB_reviews_Sentiment_analysis_LSTM.ipynb

- Or upload/open the notebook from your local machine via File → Upload notebook, or open it from Google Drive.

Once the notebook is open in Colab, run these copy-paste-ready cells in order. The instructions below assume you upload the CSV directly into the Colab session (recommended for your workflow). An alternative Drive-based approach is provided afterwards.

1) Upload the dataset directly into the Colab session.

2) Install required Python packages in the Colab runtime.

3) enable TPU: Runtime → Change runtime type → TPU.

4) After training, save models/checkpoints to a persistent location.

## Local fallback (run on your machine)

If you prefer to run locally (less convenient for GPU access but useful for development):

1. Create and activate a Python virtual environment (Python 3.8+ recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk tensorflow tqdm
```

3. Open the notebook with Jupyter or run scripts in `src/` if present. To open the notebook locally:

```bash
jupyter lab notebooks/IMDB_reviews_Sentiment_analysis_LSTM.ipynb
```

Note: the notebook is organized top-to-bottom: data loading → preprocessing → model → training → evaluation.

## Expected results

On a reasonable preprocessing pipeline and a modest LSTM, expected validation accuracy is typically in the 85–92% range depending on hyperparameters and embeddings. The notebook reports accuracy, precision, recall, F1-score and shows a confusion matrix.

## Evaluation metrics

- Accuracy
- Precision / Recall
- F1-score
- Confusion matrix

Log hyperparameters (embedding size, LSTM units, dropout, batch size, number of epochs) and random seeds for reproducibility.

## Dataset citation

This project uses the IMDB movie reviews dataset (commonly available on Kaggle and other mirrors).