from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from utils import save_data, train_model, predict_hand_sign, NUM_LANDMARKS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Fixes OPTIONS preflight

# ========== API Endpoints ==========

@app.route("/capture", methods=["POST"])
def capture():
    content = request.json
    user_id = content.get("user_id")
    landmarks = content.get("landmarks")
    label = content.get("label")

    if not user_id or not landmarks or not label:
        return jsonify({"status": "error", "msg": "Missing user_id, landmarks, or label"}), 400

    if len(landmarks) != NUM_LANDMARKS * 3:
        return jsonify({"status": "error", "msg": "Invalid landmarks"}), 400

    save_data(user_id, [landmarks + [label]])
    return jsonify({"status": "success", "msg": f"Captured for '{label}'"})

@app.route("/train", methods=["POST"])
def train():
    user_id = request.json.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "msg": "Missing user_id"}), 400

    path, msg = train_model(user_id)
    if path:
        return jsonify({"status": "success", "msg": msg})
    else:
        return jsonify({"status": "error", "msg": msg}), 400

@app.route("/predict", methods=["POST"])
def predict():
    content = request.json
    user_id = content.get("user_id")
    landmarks = content.get("landmarks")

    if not user_id or not landmarks:
        return jsonify({"status": "error", "msg": "Missing user_id or landmarks"}), 400

    pred_text, confidence = predict_hand_sign(user_id, landmarks)
    if pred_text:
        return jsonify({"status": "success", "prediction": pred_text})
    else:
        return jsonify({"status": "error", "msg": confidence}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    
