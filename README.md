#  Twitter Sentiment Analysis – Real-Time Pipeline

## Description

This project is a **complete real-time sentiment analysis pipeline** for tweets, developed using **Python**, **Streamlit**, and **Machine Learning**.  
It covers every stage from data streaming simulation to sentiment prediction and dynamic dashboard visualization.

This end-to-end system was created as a challenge to integrate **natural language processing (NLP)**, **real-time data flow**, and **user interface design** into one complete data science product.

---

## Dataset

We used the [Sentiment140 dataset from Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140), containing **1.6 million tweets** labeled as **positive**, **negative**, or **neutral**.

Dataset file: `training.1600000.processed.noemoticon.csv`

---

## Features

✅ **Tweet Streaming Simulation**  
Simulates a real-time tweet stream using the Sentiment140 CSV dataset.

✅ **Text Preprocessing**  
Cleans and normalizes tweet text (lowercasing, stopword removal, emoji/URL/user filtering).

✅ **TF-IDF Vectorization**  
Applies TF-IDF with n-gram support, stopwords filtering, and document frequency thresholds.

✅ **ML Model Training & Selection**  
Trains and evaluates multiple models:
- Logistic Regression  
- Naive Bayes  
- SVM  
- SGD Classifier  
- Passive-Aggressive Classifier  
- Random Forest  

The best model is selected using **GridSearchCV** and saved as `final_model.pkl`.

✅ **Model Persistence**  
Best-performing model and vectorizer are saved using `joblib` for efficient real-time inference.

✅ **Interactive Streamlit Dashboard**  
Real-time web app that displays:
- Global sentiment statistics  
- Pie & bar charts  
- Timeline of sentiment over time  
- Recent tweets with predictions  
- Auto-refresh every X seconds  
- Custom background image and dark theme  

---
