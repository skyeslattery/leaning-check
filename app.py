from flask import Flask, render_template, request, redirect, url_for, flash, session
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from logging.handlers import RotatingFileHandler
from werkzeug.exceptions import HTTPException
from models import db, TextAnalysis
from datetime import datetime

app = Flask(__name__)
app.secret_key = '##########'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///political_text_analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

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
    categories = ['Left', 'Center', 'Right']
    return categories[probabilities.index(max(probabilities))], max(probabilities)

@app.route('/')
def home():
    history = TextAnalysis.query.order_by(TextAnalysis.timestamp.desc()).limit(10).all()
    return render_template('home.html', history=history)

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    if not text:
        flash('Please enter some text to analyze.')
        return redirect(url_for('home'))

    try:
        result, probability = analyze_text(text)
        new_analysis = TextAnalysis(text=text, result=result, probability=probability, timestamp=datetime.utcnow())
        db.session.add(new_analysis)
        db.session.commit()
    except Exception as e:
        app.logger.error(f"Error analyzing text: {e}")
        flash('An error occurred while analyzing the text. Please try again later.')
        return redirect(url_for('home'))

    return redirect(url_for('result', analysis_id=new_analysis.id))

@app.route('/result/<int:analysis_id>')
def result(analysis_id):
    analysis = TextAnalysis.query.get_or_404(analysis_id)
    return render_template('result.html', result=analysis.result, text=analysis.text, probability=analysis.probability)

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

@app.route('/delete/<int:analysis_id>', methods=['POST'])
def delete_analysis(analysis_id):
    analysis = TextAnalysis.query.get_or_404(analysis_id)
    try:
        db.session.delete(analysis)
        db.session.commit()
    except Exception as e:
        app.logger.error(f"Error deleting analysis: {e}")
        flash('An error occurred while deleting the analysis entry.')
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  
    app.run(debug=True)

    
