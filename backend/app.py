import os
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder="../results")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # backend/
RESULTS_DIR = os.path.join(BASE_DIR, "../results")
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")

@app.route('/')
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")
import json

@app.route('/metrics')
def get_metrics():
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)
    return jsonify(metrics)


@app.route('/static/<path:filename>')
def static_files(results):
    return send_from_directory(RESULTS_DIR, results)

if __name__ == '__main__':
    app.run(debug=True)
