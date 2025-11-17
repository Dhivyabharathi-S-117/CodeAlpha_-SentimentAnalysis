import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from nltk.corpus import stopwords
import string

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')

# -----------------------------------------
# STEP 1: Load Dataset
# -----------------------------------------
try:
    df = pd.read_csv('YoutubeCommentsDataSet.csv')
    print("✅ Dataset loaded successfully:", df.shape)
except Exception as e:
    print("⚠️ Could not load file, using sample data.")
    df = pd.DataFrame({
        'comment': [
            "I love this video! So inspiring!",
            "This is the worst tutorial ever.",
            "Not bad, pretty decent.",
            "Amazing work, keep it up!",
            "Terrible sound quality, disappointed.",
            "Nice explanation but a bit long.",
            "Loved it! Learned a lot.",
            "Totally useless and boring.",
            "Good effort, but could be better.",
            "Excellent editing and clear voice!"
        ],
        'sentiment': ['Positive','Negative','Neutral','Positive','Negative','Neutral','Positive','Negative','Neutral','Positive']
    })
    print("⚠️ Sample dataset created:", df.shape)

# Keep only useful columns
df = df[['comment', 'sentiment']]

# -----------------------------------------
# STEP 2: Clean the Text
# -----------------------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['cleaned'] = df['comment'].apply(clean_text)

# -----------------------------------------
# STEP 3: Sentiment Analysis (VADER)
# -----------------------------------------
sia = SentimentIntensityAnalyzer()
df['vader_score'] = df['cleaned'].apply(lambda x: sia.polarity_scores(x)['compound'])

def classify(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['predicted_sentiment'] = df['vader_score'].apply(classify)
# ----------------------------
# STEP: Evaluation (accuracy, report, confusion matrix)
# ----------------------------

# Normalize case to avoid mismatch issues
# (convert actual labels in dataset and predicted labels to lowercase)
df['sentiment'] = df['sentiment'].astype(str).str.lower()                # actual label column
df['predicted_sentiment'] = df['predicted_sentiment'].astype(str).str.lower()

# If your dataset uses different label words, map them to standard: 'positive','negative','neutral'
# Example mapping (uncomment and edit if needed):
# mapping = {'pos':'positive','neg':'negative','neu':'neutral','0':'negative','4':'positive'}
# df['sentiment'] = df['sentiment'].replace(mapping)
# df['predicted_sentiment'] = df['predicted_sentiment'].replace(mapping)

# Remove rows where either label is missing (optional but safer)
df_eval = df[ df['sentiment'].notna() & df['predicted_sentiment'].notna() ].copy()

# Compute accuracy and print
accuracy = accuracy_score(df_eval['sentiment'], df_eval['predicted_sentiment'])
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(df_eval['sentiment'], df_eval['predicted_sentiment'],zero_division=0))

# Confusion matrix (raw)
cm = confusion_matrix(df_eval['sentiment'], df_eval['predicted_sentiment'],
                      labels=['positive','neutral','negative'])
print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

# Plot confusion matrix heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['positive','neutral','negative'],
            yticklabels=['positive','neutral','negative'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

df_eval.to_csv('youtube_sentiment_with_predictions.csv', index=False)
print("\n✅ Saved evaluated results to youtube_sentiment_with_predictions.csv")

# -----------------------------------------
# STEP 4: Emotion Detection (NRCLex)
# -----------------------------------------

# Find the dominant (strongest) emotion in each comment
def get_dominant_emotion(text):
    emotion=NRCLex(text)
    if emotion.top_emotions:
        return emotion.top_emotions[0][0]
    return "none"

df['sentiments'] = df['cleaned'].apply(get_dominant_emotion)

# -----------------------------------------
# STEP 5: Polarity & Subjectivity (TextBlob)
# -----------------------------------------
df['polarity'] = df['cleaned'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['cleaned'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# -----------------------------------------
# STEP 6: Visualizations
# -----------------------------------------
sns.set(style="whitegrid")

# Sentiment Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='predicted_sentiment',hue='predicted_sentiment', data=df, palette='cool')
plt.title("Sentiment Distribution (YouTube Comments)")
plt.xlabel("Sentiments")
plt.ylabel("Number of Comments")
plt.tight_layout()
plt.show()

# -----------------------------------------
# EXTRA: Emotion Distribution Visualization
# -----------------------------------------
plt.figure(figsize=(8,5))
sns.countplot(x='sentiments',hue='sentiments', data=df, palette='Spectral',legend= False)
plt.title("Dominant Emotions in YouTube Comments")
plt.xlabel("Emotion")
plt.ylabel("Number of Comments")
plt.show()

# -----------------------------------------
# STEP 7: Summary Results
# -----------------------------------------
print("\n📊 Sentiment Summary:")
print(df['predicted_sentiment'].value_counts())

print("\n💬 Sample Results:")
print(df[['comment', 'predicted_sentiment', 'sentiments']].head(10))

# Save to CSV
df.to_csv('youtube_sentiment_with_emotions.csv', index=False)
print("\n✅ Results (with emotions) saved to youtube_sentiment_with_emotions.csv")

