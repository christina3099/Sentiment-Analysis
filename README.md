# Movie Review Sentiment Analysis

This project explores sentiment classification on movie reviews using a variety of classical machine learning and deep learning models. The goal is to compare different text representation and modeling strategies, analyze their trade-offs, and build an efficient sentiment analysis pipeline.

## 🚀 Project Overview

Objective: Predict whether a given movie review expresses a positive or negative sentiment.

Dataset: Movie Reviews Dataset from Kaggle.

## Models Tested:

Classical ML

Bag-of-Words + Logistic Regression

TF-IDF + Logistic Regression

Neural Networks

Embedding Layer + RNN

Embedding Layer + LSTM / BiLSTM

Embedding Layer + GRU

Pretrained Embeddings

GloVe/Word2Vec + LSTM


 # Methodology

<img width="404" height="210" alt="image" src="https://github.com/user-attachments/assets/f078c828-e1a8-4417-96af-d75ad109e484" />


Data Preprocessing

Tokenization, lowercasing, punctuation/URL removal.

Stopword removal + Lemmatization.

Padding sequences for neural networks.

Vectorization

Bag-of-Words, TF-IDF for classical models.

Learned embeddings and pretrained embeddings for neural models.

# Model Training & Evaluation

Accuracy, F1-score (macro), training time recorded for comparison.

Early stopping used to prevent overfitting in deep models.

<img width="463" height="463" alt="image" src="https://github.com/user-attachments/assets/c9510b3a-f040-4634-a8d6-3f2149256e31" />

# Trade-offs

Classical ML (TF-IDF + LR): Fast, interpretable, strong performance → best choice for this dataset.

Deep Learning Models: Require more computation, hyperparameter tuning, and larger datasets to outperform classical methods.

Pretrained Embeddings: Useful for semantic understanding but need fine-tuning to be effective.

Transformers (Future Work): Expected to outperform traditional baselines once fine-tuned.

# Tech Stack

Python, NumPy, Pandas, Matplotlib, Seaborn

Scikit-learn (vectorization + logistic regression)

TensorFlow / Keras (RNN, LSTM, GRU, BiLSTM)

Hugging Face Transformers (DistilBERT)

# 📌 Next Steps

Fine-tune DistilBERT/BERT for improved accuracy.

Explore regularization and dropout in BiLSTM models.

Perform hyperparameter tuning for embedding-based architectures.

