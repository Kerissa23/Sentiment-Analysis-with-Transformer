from flask import Flask, render_template, request
from infer import predict_sentiment

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    score = None
    if request.method == 'POST':
        text = request.form['text']
        sentiment, score = predict_sentiment(text)
    return render_template('index.html', sentiment=sentiment, score=score)

if __name__ == '__main__':
    app.run(debug=False)
