from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class TextAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(10), nullable=False)  
    timestamp = db.Column(db.DateTime, nullable=False)
    probability = db.Column(db.Float, nullable=False)  

