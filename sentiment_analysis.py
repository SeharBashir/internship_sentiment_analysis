# sentiment_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1️⃣ Load dataset
df = pd.read_csv("data/feedback.csv")
print("Dataset shape:", df.shape)
print(df.head())

# 2️⃣ Preprocess
df.dropna(subset=["feedback", "label"], inplace=True)

# 3️⃣ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    df["feedback"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 4️⃣ Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5️⃣ Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 6️⃣ Evaluate
y_pred = model.predict(X_test_tfidf)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 7️⃣ Save model and vectorizer
import joblib
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved in 'models/' folder.")
