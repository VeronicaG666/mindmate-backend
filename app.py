from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import requests
import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure DB
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
db = SQLAlchemy(app)

HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")

# JournalEntry model
class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    mood = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"message": "Flask backend + DB is working!"})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    print("🧠 /api/analyze endpoint was hit")
    data = request.get_json()
    journal_text = data.get('text', '')

    if not journal_text.strip():
        return jsonify({'error': 'No text provided'}), 400

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
    }

    response = requests.post(
        "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment",
        headers=headers,
        json={"inputs": journal_text}
    )

    print("Status Code:", response.status_code)
    print("Response Body:", response.text)

    try:
        result = response.json()
        if isinstance(result, dict) and "error" in result:
            return jsonify({'error': result["error"]}), 500

        label_map = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive"
        }

        top_prediction = result[0][0]
        raw_label = top_prediction['label']
        score = top_prediction['score']
        label = label_map.get(raw_label, "unknown")

        # Save to DB
        entry = JournalEntry(text=journal_text, mood=label, confidence=round(score, 2))
        db.session.add(entry)
        db.session.commit()

        return jsonify({
            'mood': label,
            'confidence': round(score, 2)
        })

    except Exception as e:
        print("Exception:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=True)

