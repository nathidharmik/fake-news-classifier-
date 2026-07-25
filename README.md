# Fake News Detection using Machine Learning

Classifies news articles as **Real** or **Fake** using TF-IDF vectorization and Logistic Regression, trained on the ISOT Fake and Real News Dataset.

## Overview

Fake news spreads fast online. This project applies NLP + ML to automatically flag news articles as real or fake, converting text into numerical features with TF-IDF and classifying with Logistic Regression.

## Features

- Real vs. fake classification
- Text cleaning & preprocessing
- TF-IDF feature extraction
- Evaluation via accuracy, confusion matrix, classification report
- Predictions on custom input text

## Tech Stack

Python 3 · Jupyter Notebook · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

## Dataset

**ISOT Fake and Real News Dataset** — `Fake.csv` and `True.csv`, each with `title`, `text`, `subject`, `date`.

## Pipeline

Import data → clean & preprocess text → train-test split → TF-IDF vectorization → train Logistic Regression → evaluate → predict on new articles.

## Results

| Metric | Score |
|---|---|
| Training Accuracy | 99.10% |
| Testing Accuracy | 98.46% |

Confusion Matrix: TP 4239 · TN 4704 · FP 6 · FN 31

## Project Structure

```
Fake-News-Detection/
├── Fake.csv
├── True.csv
├── fake_news_classifier.ipynb
├── README.md
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection
pip install -r requirements.txt
jupyter notebook
```

Open `fake_news_classifier.ipynb` and run all cells.

## Example

**Input:** "The Reserve Bank of India kept its benchmark interest rate unchanged during the latest monetary policy meeting."
**Output:** `REAL NEWS`

## Future Work

- Streamlit/Flask deployment
- Deep learning (LSTM/BERT)
- Live URL-based detection

## Author

**Dharmik N** — B.Tech CSE (AI & ML), Raghu Engineering College
