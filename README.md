# VADER-Sentiment-Analysis

Simple NLTK Python project for analyzing sentiment behind a customer review. 
Use nltk.download('vader_lexicon') to download VADER Lexicon

Reviews.csv contains reviews from Amazon customers.

# cURL download
 #!/bin/bash
curl -L -o ~/Downloads/amazon-fine-food-reviews.zip\
  https://www.kaggle.com/api/v1/datasets/download/snap/amazon-fine-food-reviews

Tokenizes comment text and uses SentimentIntensityAnalyzer to classify text as negative, neutral, or positive.

CComment is decided to be positive/negative based on overall score being over/under +-0.3.
