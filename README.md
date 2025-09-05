 Task 2: Sentiment Analysis

# Project Description
This project implements a simple **sentiment analysis model** that classifies IMDB movie reviews as **positive** or **negative**.  
The goal is to build a baseline machine learning model using text preprocessing, feature extraction, and logistic regression.

# Steps

1. Preprocessing
   - Convert text to lowercase.
   - Remove punctuation, numbers, and stopwords.
   - Clean review text for better accuracy.

2. Feature Engineering
   - Use *CountVectorizer* to convert text into numerical features (bag-of-words representation).

3. Model Training
   - Train a *Logistic Regression** model to classify reviews.

4. Evaluation
   - Model achieved around **87% accuracy** on the IMDB dataset.
   - Optional: Precision, recall, F1-score, and confusion matrix can also be computed.

5. Prediction
   - The script predicts whether a new review is *positive* or *negative*.


# Files
- `sentiment_analysis.py` → Main Python script.
- `imdb_reviews_sample.csv` → Small dataset for testing.
- `README.md` → Project documentation.


# How to Run

1. Upload dataset in Colab or keep it in the same folder as script.
   ```python
   df = pd.read_csv("imdb_reviews_sample.csv")
   ```

2. Run the script:
   ```bash
   python sentiment_analysis.py
   ```

3. Example output:
   ```
    Model Accuracy: 87.00 %
   Review Sentiment: Positive
   ```

---

#Example Prediction
```python
sample_review = "The movie was absolutely wonderful, I loved it!"
```
Output:
```
 Review Sentiment: Positive
```

---

#Dataset
- Small sample dataset is included in this repo: `imdb_reviews_sample.csv`
- Full dataset (50,000 IMDB reviews) can be downloaded from Kaggle:  
  [IMDB Movie Reviews Dataset (50K)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

#Outcome
A working **Python script** that:
- Trains a sentiment analysis model.
- Evaluates accuracy.
- Predicts sentiment of new reviews (Positive/Negative).
