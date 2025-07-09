# FakeNews-Detector

## 🔍 Overview
This project uses Natural Language Processing (NLP) and machine learning to identify whether a given news article is **real** or **fake**. It uses TF-IDF for vectorizing text data and a Passive Aggressive Classifier for training and prediction.

## 🗂️ Dataset
- File: `FakeNewsNet.csv`
- Contains URLs and labels (`real` or `fake`)
- You can download it from [https://www.kaggle.com/datasets/algord/fake-news]

## ✨ Features
- Cleaned and preprocessed news dataset
- TF-IDF Vectorization to convert text to numerical format
- Passive Aggressive Classifier for classification
- Accuracy, Confusion Matrix, and Classification Report for evaluation

## ⚙️ Installation
1. **Clone the repository:**
```bash
git clone https://github.com/Snigdha604/FakeNews-Detector.git
cd FakeNews-Detector
```

## Install dependencies:
```bash
pip install -r requirements.txt
```

## Sample Output
🔹 Accuracy: 0.9273

🔹 Confusion Matrix:
[[1220   34]
 [  71 1134]]

🔹 Classification Report:
              precision    recall  f1-score   support

       fake       0.94      0.97      0.95      1254
       real       0.97      0.94      0.95      1205

    accuracy                           0.95      2459
   macro avg       0.95      0.95      0.95      2459
weighted avg       0.95      0.95      0.95      2459

## 🛠️ Technologies Used
Python
Pandas
Scikit-learn
TF-IDF Vectorizer
Passive Aggressive Classifier

## 📄 License
This project is licensed under the MIT License.

