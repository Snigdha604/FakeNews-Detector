# fake_news_detector.py

# 📦 Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 📂 Load the dataset
df = pd.read_csv('FakeNewsNet.csv')

# 🧹 Inspect and clean the dataset
print("First 5 rows:\n", df.head())
print("\nColumns available:", df.columns)
print("\nNull values:\n", df.isnull().sum())

# ✏️ Rename and drop missing values
df = df.rename(columns={'news_url': 'text', 'real': 'label'})
df = df.dropna(subset=['text', 'label'])

# 🧠 Split the data into training and test sets
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 🔠 TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# 🚀 Train Passive Aggressive Classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# 🎯 Make predictions
y_pred = model.predict(tfidf_test)

# 🧾 Convert to string for evaluation
y_test = y_test.astype(str)
y_pred = [str(label) for label in y_pred]

# 📊 Evaluation Metrics
print("\n✅ Evaluation Metrics:")

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n🔹 Accuracy: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🔹 Confusion Matrix:")
print(cm)

# Classification Report
cr = classification_report(y_test, y_pred, zero_division=0)
print("\n🔹 Classification Report:")
print(cr)
