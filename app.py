from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)
sid = SentimentIntensityAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        scores = sid.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        result = {'text': text, 'sentiment': sentiment, 'score': compound}
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
