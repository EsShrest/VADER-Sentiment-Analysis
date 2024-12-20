import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon') 


def determine_sentiment(compound): # Function to determine sentiment based on compound score from SentimentIntensityAnalyzer
    if compound >= 0.3:
        return 'good'
    elif compound <= -0.3:
        return 'bad'
    else:
        return 'neutral'


sia = SentimentIntensityAnalyzer()

df = pd.read_csv('Reviews.csv') # Loading Amazon reviews dataset
print(df.head())

results = {}

for index, row in df.iterrows():
    Id = row['Id']
    text = row['Text']
    results[Id] = sia.polarity_scores(text)

Final = pd.DataFrame(results).T 
Final['Id'] = Final.index
Final = pd.merge(df, Final, on='Id')


Final['Sentiment'] = Final['compound'].apply(determine_sentiment)
print(Final.head())

# Plotting the sentiment distribution
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=Final, x='Score', y='pos', ax=axs[0])
sns.barplot(data=Final, x='Score', y='neu', ax=axs[1])
sns.barplot(data=Final, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()