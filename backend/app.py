"""
Farmer Crop Disease Detection - Flask Web App
------------------------------------------------
Loads a trained Keras model and serves a simple web UI where a user
(farmer / agronomist) can upload a photo of a plant leaf and get an
instant disease prediction along with a short description and
recommended action.

Run:
    python app.py

Make sure you have trained a model first (see model/train_model.py)
and that model/plant_disease_model.h5 and model/class_names.json exist.
"""

import os
import json
import uuid
from datetime import datetime

import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

from utils.preprocess import load_and_preprocess_image
from utils.disease_info import DISEASE_INFO

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_disease_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB max upload
app.secret_key = "change-this-secret-key-in-production"

# ------------------------------------------------------------------
# Load model + class names once at startup
# ------------------------------------------------------------------
model = None
class_names = []
model_load_error = None

try:
    import tensorflow as tf

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model_load_error = (
            "No trained model found at model/plant_disease_model.h5. "
            "Run `python model/train_model.py` first (see README)."
        )

    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
except Exception as e:  # pragma: no cover
    model_load_error = f"Error loading model: {e}"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_ready=model is not None, error=model_load_error)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        flash(model_load_error or "Model is not loaded yet.")
        return redirect(url_for("index"))

    if "file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PNG or JPG image.")
        return redirect(url_for("index"))

    # Save upload with a unique name
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    # Preprocess + predict
    img_array = load_and_preprocess_image(save_path, target_size=IMG_SIZE)
    preds = model.predict(img_array)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100

    raw_label = class_names[top_idx] if class_names else f"class_{top_idx}"
    info = DISEASE_INFO.get(raw_label, {
        "crop": raw_label.split("___")[0].replace("_", " "),
        "disease": raw_label.split("___")[-1].replace("_", " ") if "___" in raw_label else raw_label,
        "description": "No additional information available for this class.",
        "recommendation": "Consult a local agricultural extension officer for confirmation and treatment options."
    })

    # Top-3 predictions for extra context
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [
        {
            "label": (class_names[i] if class_names else f"class_{i}").replace("___", " - ").replace("_", " "),
            "confidence": round(float(preds[i]) * 100, 2),
        }
        for i in top3_idx
    ]

    result = {
        "image_path": url_for("static", filename=f"uploads/{unique_name}"),
        "crop": info["crop"],
        "disease": info["disease"],
        "is_healthy": "healthy" in raw_label.lower(),
        "confidence": round(confidence, 2),
        "description": info["description"],
        "recommendation": info["recommendation"],
        "top3": top3,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    return render_template("result.html", result=result)


@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
