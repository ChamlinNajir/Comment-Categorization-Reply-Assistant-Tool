# NLP Comment Categorization & Reply Assistant

## Project Overview
This tool analyzes user comments using Natural Language Processing (NLP) to classify them into categories like Praise, Support, Constructive Criticism, and Hate/Abuse. It helps community managers respond efficiently by suggesting context-aware replies.

## Features
- **Text Cleaning:** Removes noise, stopwords, and performs lemmatization.
- **Classification:** Uses Logistic Regression with TF-IDF Vectorization.
- **Hybrid Logic:** Combines Machine Learning with Keyword Overrides for high accuracy on edge cases.
- **Interactive UI:** A Streamlit web app for real-time testing.

## Tech Stack
- **Python 3.x**
- **Libraries:** Pandas, NLTK, Scikit-learn, Streamlit
- **Algorithm:** Logistic Regression

## How to Run
1. Install dependencies:
    pip install pandas scikit-learn nltk streamlit

2. Train the model (if .pkl files are missing):
    python train_model.py

3. Run the App:
    python -m streamlit run app.py


## Project Structure
- train_model.py: Trains and saves the model.

- app.py: The main user interface.

- preprocessing.py: Contains text cleaning logic.