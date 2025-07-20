from transformers import pipeline

# Disable TensorFlow by forcing PyTorch use
sentiment_pipeline = pipeline("sentiment-analysis", framework="pt")

def predict_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']
