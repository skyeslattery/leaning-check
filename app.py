from flask import Flask, render_template, request, redirect, url_for, flash, session
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from logging.handlers import RotatingFileHandler
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")

def analyze_text(text):
    # Truncate to maximum token length (512)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = logits.softmax(dim=-1)[0].tolist()
    categories = ['left', 'center', 'right']
    return categories[probabilities.index(max(probabilities))]

@app.route('/')
def home():
    history = session.get('history', [])
    history.reverse()
    return render_template('home.html', history=history)

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    if not text:
        flash('Please enter some text to analyze.')
        return redirect(url_for('home'))

    try:
        result = analyze_text(text)
    except Exception as e:
        app.logger.error(f"Error analyzing text: {e}")
        flash('An error occurred while analyzing the text. Please try again later.')
        return redirect(url_for('home'))

    history = session.get('history', [])
    history.append({'text': text, 'result': result})
    session['history'] = history

    return redirect(url_for('result', index=len(history)-1))

@app.route('/result/<int:index>')
def result(index):
    history = session.get('history', [])
    if index < 0 or index >= len(history):
        flash('Invalid history index.')
        return redirect(url_for('home'))

    entry = history[index]
    return render_template('result.html', result=entry['result'], text=entry['text'], index=index)


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    app.logger.error(f"HTTP exception occurred: {e}")
    flash('An unexpected error occurred. Please try again later.')
    return redirect(url_for('home'))

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unexpected error: {e}")
    flash('An unexpected error occurred. Please try again later.')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
