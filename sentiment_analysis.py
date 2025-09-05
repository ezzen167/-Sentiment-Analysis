# sentiment_analysis.py
# Task 2: Sentiment Analysis

import pandas as pd
import re, string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import nltk

# Download stopwords if running first time
nltk.download("stopwords")

# ==========================
# Step 1: Load Dataset
# ==========================
# Use sample dataset for GitHub testing
df = pd.read_csv("imdb_reviews_sample.csv")   # change filename to imdb_reviews.csv for full dataset

print("Dataset shape:", df.shape)
print(df.head())

# ==========================
# Step 2: Preprocessing
# ==========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)                               # remove numbers
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]              # remove stopwords
    return " ".join(words)

df["cleaned_review"] = df["review"].apply(clean_text)

# ==========================
# Step 3: Features & Labels
# ==========================
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)

# ==========================
# Step 4: Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# Step 5: Train Model
# ==========================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ==========================
# Step 6: Evaluation
# ==========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", round(acc * 100, 2), "%")

# ==========================
# Step 7: Example Prediction
# ==========================
sample_review = "The movie was absolutely wonderful, I loved it!"
clean_sample = clean_text(sample_review)
sample_vector = vectorizer.transform([clean_sample])
prediction = model.predict(sample_vector)

print("🔮 Sample Review Prediction:", "Positive" if prediction[0] == 1 else "Negative")
