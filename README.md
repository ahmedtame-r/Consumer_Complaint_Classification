# Consumer Complaint Classification using NLP & Deep Learning

## 📌 Overview

This project develops an end-to-end Natural Language Processing (NLP) system that automatically classifies customer complaint narratives into their corresponding product categories.

The project compares three recurrent neural network architectures (**SimpleRNN, LSTM, and GRU**) with a fine-tuned **DistilBERT Transformer** to evaluate the effectiveness of traditional sequence models versus modern Transformer-based language models.

The best-performing model is deployed using **Gradio**, providing an interactive web interface for real-time complaint classification.

---

## 🎯 Objectives

* Build a complete NLP text classification pipeline.
* Perform text preprocessing and feature preparation.
* Train multiple deep learning models.
* Compare model performance using standard evaluation metrics.
* Deploy the best model for inference.

---

## 📂 Project Pipeline

### 1. Data Exploration

* Dataset inspection
* Missing value analysis
* Class distribution visualization

### 2. Text Preprocessing

* Lowercasing
* Removing punctuation
* Removing stopwords
* Lemmatization
* Text cleaning

> DistilBERT uses its own tokenizer and therefore receives the raw complaint text.

### 3. Data Preparation

* Train/Test split
* Stratified sampling
* Class imbalance handling

### 4. Tokenization

* Vocabulary generation
* Sequence padding
* Input preparation

### 5. Model Development

The following models were implemented and compared:

* ✅ SimpleRNN
* ✅ LSTM
* ✅ GRU
* ✅ Fine-tuned DistilBERT

---

## 📊 Evaluation Metrics

Each model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

The comparison highlights the advantages of Transformer-based models over traditional recurrent neural networks for complaint classification.

---

## 🚀 Deployment

The best-performing model is deployed using **Gradio**, enabling users to enter complaint text and instantly receive the predicted complaint category.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Hugging Face Transformers
* DistilBERT
* Gradio
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 📁 Project Structure

```text
Consumer_Complaint_Classification/
│
├── notebooks/
│   ├── Data_Preprocessing.ipynb
│   ├── Model_Training.ipynb
│   ├── Evaluation.ipynb
│   └── Deployment.ipynb
│
├── saved_models/
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

---

## 💡 Key Features

* End-to-end NLP pipeline
* Multiple deep learning architectures
* Transformer vs RNN comparison
* Comprehensive model evaluation
* Interactive deployment with Gradio
* Clean and reproducible workflow

---

## 📈 Future Improvements

* Experiment with larger Transformer models (RoBERTa, DeBERTa).
* Hyperparameter optimization.
* Data augmentation for minority classes.
* Explainable AI (LIME/SHAP).
* Docker deployment.
* Cloud deployment (Hugging Face Spaces or Streamlit Cloud).

---

## ⭐ Acknowledgements

This project was developed for educational and research purposes to explore modern NLP techniques for automated complaint classification using deep learning and Transformer models.
