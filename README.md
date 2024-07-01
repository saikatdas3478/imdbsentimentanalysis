# IMDB Reviews Sentiment Analysis

## Overview
This project demonstrates sentiment analysis on the IMDB movie reviews dataset using various NLP techniques and machine learning models to classify reviews as positive or negative. IMDb (an acronym for Internet Movie Database) is an online database of information related to films, television series, podcasts, home videos, video games, and streaming content online including cast, production crew and personal biographies, plot summaries, trivia, ratings, and fan and critical reviews. This document is about training a machine learning model to flag sentiment using IMDb dataset. The document includes steps for data preprocessing, tokenizing, pos-tagging, lemmatizing, and sentiment analysis. 
It also compares the results of VaderSentiment libraries and human level labeling for sentiment analysis. The document presents the accuracy scores, confusion matrices, and classification reports for both Naive Bayes and Random Forest algorithms applied to the dataset. Finally, it provides a function to predict sentiment based on input text. Any kind of negative or positive review will be flagged correctly with the use of this model.

## Dataset
- **Source**: 50,000 IMDB movie reviews.
- **Sample**: 10,000 randomly selected reviews.

## Methodology

### Data Preprocessing
1. **Text Cleaning**: Removing HTML tags, punctuations, and junk characters.
2. **Tokenization and Lemmatization**: Tokenizing, POS tagging, and lemmatizing text.
3. **Sentiment Validation**: Using VADER sentiment analysis to check label validity.

### Feature Extraction
1. **Count Vectorization**: Converting text to a bag-of-words model.
2. **TF-IDF Vectorization**: Converting text to a TF-IDF model.

### Model Training and Evaluation
1. **Naive Bayes**: Using Gaussian Naive Bayes for classification.
2. **Random Forest**: Using Random Forest for classification.
3. **Evaluation Metrics**: Accuracy, Confusion Matrix, and Classification Report.

## Results
- **Naive Bayes**:
  - Count Vectorizer Accuracy: 63.45%
  - TF-IDF Vectorizer Accuracy: 62.6%
- **Random Forest**:
  - Count Vectorizer Accuracy: 82.95%
  - TF-IDF Vectorizer Accuracy: 81.85%

## How to Run the Code
1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    ```
2. **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the script**:
    ```bash
    python sentiment_analysis.py
    ```

## Conclusion
Random Forest Classifier with Count Vectorization provides the best performance in classifying the sentiment of movie reviews.

## Sample Usage
```python
neg_text = '''My biggest disappointment with this phone is the UI. ... '''
print(output_sentiment(neg_text))  # Output: Negative

pos_text = '''I recently purchased a new phone, and I must say ... '''
print(output_sentiment(pos_text))  # Output: Positive
