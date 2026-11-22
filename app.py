import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, jsonify, render_template, request


app = Flask(__name__)
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/api/extract", methods=["POST"])
def extract():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        api_request = Request(
            f"{FASTAPI_URL.rstrip('/')}/extract",
            data=json.dumps({
                "text": text,
                "min_score": data.get("min_score", 0.0),
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(api_request, timeout=120) as api_response:
            payload = json.loads(api_response.read().decode("utf-8"))
        return jsonify(payload)
    except HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"error": f"NER API returned HTTP {error.code}"}
        return jsonify(payload), error.code
    except URLError:
        return jsonify({
            "error": "The NER API is not running. Start it with: uvicorn api.main:app --reload"
        }), 503

@app.route("/api/redact", methods=["POST"])
def redact():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    style = data.get("style", "label")

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if style not in ("label", "mask"):
        return jsonify({"error": "Style must be label or mask"}), 400

    try:
        api_request = Request(
            f"{FASTAPI_URL.rstrip('/')}/redact",
            data=json.dumps({
                "text": text,
                "style": style,
                "placeholder": "[{type}]",
                "min_score": data.get("min_score", 0.0),
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(api_request, timeout=120) as api_response:
            payload = json.loads(api_response.read().decode("utf-8"))
        return jsonify(payload)
    except HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"error": f"NER API returned HTTP {error.code}"}
        return jsonify(payload), error.code
    except URLError:
        return jsonify({
            "error": "The NER API is not running. Start it with: uvicorn api.main:app --reload"
        }), 503
    
if __name__ == "__main__":
    app.run(debug=True)
