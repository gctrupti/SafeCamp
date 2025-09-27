import os
import json
from flask import Flask, render_template, jsonify
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------- CONFIG ----------------
load_dotenv()  # Load API key from .env
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("⚠️ GEMINI_API_KEY not found. Add it to your .env file.")

genai.configure(api_key=API_KEY)

app = Flask(__name__)

ALERTS_FILE = "alerts.json"
ALERTS_FOLDER = "alerts"

# ---------------- GENAI HELPER ----------------
def generate_alert_message(alert):
    """
    Generate a human-readable alert message using Gemini.
    """
    prompt = f"""
    You are a security assistant. Based on this alert JSON, generate a short alert message.
    JSON: {json.dumps(alert, indent=2)}

    Example output: 
    '⚠️ Fall detected at 14:23 near the entrance. Snapshot saved.'
    """
    try:
        # Use a supported model for v1beta
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Alert generated, but AI message failed: {str(e)}"

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/latest-alert")
def latest_alert():
    """Return the latest alert JSON + AI generated message + snapshot path."""
    if not os.path.exists(ALERTS_FILE) or os.path.getsize(ALERTS_FILE) == 0:
        return jsonify({"message": "✅ No alerts yet", "image": None})

    try:
        with open(ALERTS_FILE, "r") as f:
            alerts = json.load(f)

        if not alerts:
            return jsonify({"message": "✅ No alerts yet", "image": None})

        latest = alerts[-1]  # Most recent alert
        message = generate_alert_message(latest)

        return jsonify({
            "message": message,
            "json": latest,
            "image": latest.get("image", None)
        })

    except Exception as e:
        return jsonify({"message": f"⚠️ Error reading alerts: {str(e)}", "image": None})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
