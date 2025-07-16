from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import requests
import os
from dotenv import load_dotenv
from datetime import datetime

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

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    mood = data.get("mood", "").lower()

    RESOURCE_MAP = {
        "positive": [
            {"title": "Keep the momentum going", "url": "https://youtu.be/oTugjssqOT0"},
            {"title": "Boost your productivity", "url": "https://jamesclear.com/atomic-habits"}
        ],
        "neutral": [
            {"title": "Reset with mindfulness", "url": "https://www.youtube.com/watch?v=inpok4MKVLM"},
            {"title": "Take a creative break", "url": "https://waitbutwhy.com/"}
        ],
        "negative": [
            {"title": "You’re not alone — here’s help", "url": "https://www.youtube.com/watch?v=MIr3RsUWrdo"},
            {"title": "Talk to someone now", "url": "https://www.betterhelp.com/get-started/"}
        ]
    }

    resources = RESOURCE_MAP.get(mood, [])
    return jsonify({"resources": resources})

@app.route('/api/journals', methods=['GET'])
def get_journals():
    entries = JournalEntry.query.order_by(JournalEntry.id.desc()).limit(7).all()
    entries.reverse()

    data = [
        {
            "id": e.id,
            "text": e.text,
            "mood": e.mood,
            "confidence": e.confidence,
        }
        for e in entries
    ]

    return jsonify(data)

@app.route('/api/history', methods=['POST'])
def get_full_history():
    data = request.get_json() or {}
    session_start = data.get("sessionStart")

    try:
        # Convert to integer if valid, else default to 0
        session_id_cutoff = int(session_start)
    except (TypeError, ValueError):
        session_id_cutoff = 0

    # Fetch entries added after session started
    entries = (
        JournalEntry.query
        .filter(JournalEntry.id > session_id_cutoff)
        .order_by(JournalEntry.id.asc())
        .all()
    )

    response_data = [
        {
            "id": e.id,
            "text": e.text,
            "mood": e.mood,
            "confidence": e.confidence
        }
        for e in entries
    ]

    return jsonify(response_data)


@app.route('/api/journals/<int:entry_id>', methods=['PUT'])
def update_journal(entry_id):
    data = request.get_json()
    new_text = data.get('text')

    entry = JournalEntry.query.get(entry_id)
    if not entry:
        return jsonify({'error': 'Entry not found'}), 404

    entry.text = new_text
    db.session.commit()

    return jsonify({'message': 'Entry updated successfully'})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)
