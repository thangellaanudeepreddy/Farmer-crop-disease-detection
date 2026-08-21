"""
Farmer Crop Disease Detection - Flask Web App
"""

import os
import json
import uuid
from datetime import datetime

import numpy as np

from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    flash
)

from werkzeug.utils import secure_filename

from utils.preprocess import load_and_preprocess_image
from utils.disease_info import DISEASE_INFO


# ============================================================
# PATH CONFIGURATION
# ============================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

MODEL_DIR = os.path.join(PROJECT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

UPLOAD_FOLDER = os.path.join(
    BACKEND_DIR,
    "static",
    "uploads"
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.secret_key = "farmer-crop-disease-secret-key"


# ============================================================
# GLOBAL VARIABLES
# ============================================================

model = None
class_names = []
model_load_error = None


# ============================================================
# FALLBACK CLASS NAMES
# ============================================================
# These are the 38 PlantVillage classes used by your model.
# If class_names.json is missing/incorrect, the app can still
# convert prediction indexes such as class 26 into the real name.

DEFAULT_CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ============================================================
# LOAD TENSORFLOW + MODEL
# ============================================================

try:

    print()
    print("========================================")
    print("FARMER CROP DISEASE DETECTION")
    print("========================================")

    print("Loading TensorFlow...")

    import tensorflow as tf

    print("TensorFlow loaded successfully.")
    print("Model path:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if os.path.exists(MODEL_PATH):

        print("Model file FOUND.")

        model = tf.keras.models.load_model(MODEL_PATH)

        print("MODEL LOADED SUCCESSFULLY")

    else:

        model_load_error = (
            "Model file not found:\n" + MODEL_PATH
        )

        print("MODEL ERROR:")
        print(model_load_error)

    # --------------------------------------------------------
    # LOAD CLASS NAMES
    # --------------------------------------------------------

    if os.path.exists(CLASS_NAMES_PATH):

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            loaded_class_names = json.load(f)

        # Make sure the JSON contains a usable list.
        if isinstance(loaded_class_names, list) and len(loaded_class_names) >= 38:
            class_names = loaded_class_names
            print("Class names loaded successfully.")
            print("Number of classes:", len(class_names))
        else:
            print(
                "WARNING: class_names.json is invalid or incomplete."
            )
            print("Using built-in 38-class mapping.")
            class_names = DEFAULT_CLASS_NAMES.copy()

    else:

        print(
            "WARNING: class_names.json not found."
        )
        print("Using built-in 38-class mapping.")
        class_names = DEFAULT_CLASS_NAMES.copy()


except Exception as e:

    model_load_error = str(e)

    print()
    print("========================================")
    print("MODEL LOADING ERROR")
    print("========================================")
    print(model_load_error)

    # Even if class_names.json has a problem, keep the correct
    # model-index-to-name mapping available.
    if not class_names:
        class_names = DEFAULT_CLASS_NAMES.copy()


# ============================================================
# HELPERS
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def get_class_name(index):

    """
    Convert model output index to the real PlantVillage label.
    Example:
        26 -> Strawberry___Leaf_scorch
    """

    index = int(index)

    if 0 <= index < len(class_names):
        return class_names[index]

    if 0 <= index < len(DEFAULT_CLASS_NAMES):
        return DEFAULT_CLASS_NAMES[index]

    return f"class_{index}"


def format_label(raw_label):

    """
    Convert:
        Strawberry___Leaf_scorch
    into:
        Strawberry - Leaf scorch
    """

    if "___" in raw_label:
        crop, disease = raw_label.split("___", 1)
    else:
        crop = "Unknown"
        disease = raw_label

    crop = crop.replace("_", " ")
    disease = disease.replace("_", " ")

    return crop.strip(), disease.strip()


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def index():

    return render_template(
        "index.html",
        model_ready=(model is not None),
        error=model_load_error
    )


# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    print()
    print("========================================")
    print("PREDICT FUNCTION CALLED")
    print("========================================")

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        print("ERROR: Model is not loaded.")

        flash(
            model_load_error
            or
            "Model is not loaded."
        )

        return redirect(url_for("index"))

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if "file" not in request.files:

        print("ERROR: No file received.")

        flash("Please select a leaf image.")

        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":

        print("ERROR: Empty filename.")

        flash("Please select a leaf image.")

        return redirect(url_for("index"))

    if not allowed_file(file.filename):

        print("ERROR: Unsupported file type.")

        flash("Please upload JPG, JPEG or PNG.")

        return redirect(url_for("index"))

    # ========================================================
    # SAVE IMAGE
    # ========================================================

    try:

        filename = secure_filename(file.filename)

        extension = filename.rsplit(".", 1)[1].lower()

        unique_name = uuid.uuid4().hex + "." + extension

        os.makedirs(
            UPLOAD_FOLDER,
            exist_ok=True
        )

        save_path = os.path.join(
            UPLOAD_FOLDER,
            unique_name
        )

        file.save(save_path)

        print("Image saved:")
        print(save_path)

    except Exception as e:

        print("IMAGE SAVE ERROR:")
        print(e)

        flash("Unable to save image.")

        return redirect(url_for("index"))

    # ========================================================
    # PREPROCESS + PREDICT
    # ========================================================

    try:

        print("Preprocessing image...")

        img_array = load_and_preprocess_image(
            save_path,
            target_size=IMG_SIZE
        )

        print("Image preprocessing successful.")

        print("Running disease prediction...")

        preds = model.predict(
            img_array,
            verbose=0
        )[0]

        print("Prediction completed.")

        # ----------------------------------------------------
        # BEST PREDICTION
        # ----------------------------------------------------

        top_idx = int(np.argmax(preds))

        confidence = float(preds[top_idx]) * 100

        # ----------------------------------------------------
        # IMPORTANT:
        # Convert class 26 to its actual class name.
        # With the supplied 38-class list:
        # 26 = Strawberry___Leaf_scorch
        # ----------------------------------------------------

        raw_label = get_class_name(top_idx)

        crop_name, disease_name = format_label(raw_label)

        print("Prediction index:", top_idx)
        print("Predicted class:", raw_label)
        print("Crop:", crop_name)
        print("Disease:", disease_name)
        print("Confidence:", round(confidence, 2), "%")

        # ====================================================
        # DISEASE INFORMATION
        # ====================================================

        info = DISEASE_INFO.get(
            raw_label,
            {
                "crop": crop_name,
                "disease": disease_name,
                "description":
                    f"{disease_name} detected in {crop_name}.",
                "recommendation":
                    "Remove badly affected leaves, maintain good field hygiene, "
                    "and consult a local agricultural officer for confirmation "
                    "and suitable treatment."
            }
        )

        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        top3_idx = preds.argsort()[-3:][::-1]

        top3 = []

        for i in top3_idx:

            i = int(i)

            label = get_class_name(i)

            label_crop, label_disease = format_label(label)

            top3.append(
                {
                    "label": f"{label_crop} - {label_disease}",
                    "confidence": round(
                        float(preds[i]) * 100,
                        2
                    )
                }
            )

        # ====================================================
        # CREATE RESULT
        # ====================================================

        result = {

            "image_path": url_for(
                "static",
                filename=f"uploads/{unique_name}"
            ),

            "crop": info.get(
                "crop",
                crop_name
            ),

            "disease": info.get(
                "disease",
                disease_name
            ),

            "is_healthy":
                "healthy" in raw_label.lower(),

            "confidence": round(
                confidence,
                2
            ),

            "description": info.get(
                "description",
                "No description available."
            ),

            "recommendation": info.get(
                "recommendation",
                "Consult an agricultural officer."
            ),

            "top3": top3,

            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M"
            )
        }

        # ====================================================
        # PRINT RESULT
        # ====================================================

        print()
        print("========================================")
        print("PREDICTION COMPLETE")
        print("========================================")
        print("Crop:", result["crop"])
        print("Disease:", result["disease"])
        print("Confidence:", result["confidence"], "%")
        print("========================================")

        # ====================================================
        # RESULT PAGE
        # ========================================================

        return render_template(
            "result.html",
            result=result
        )

    # ========================================================
    # PREDICTION ERROR
    # ========================================================

    except Exception as e:

        print()
        print("========================================")
        print("PREDICTION ERROR")
        print("========================================")

        print(type(e).__name__)
        print(str(e))

        print("========================================")

        flash(
            "Unable to analyse the leaf image. "
            "Please try another clear leaf image."
        )

        return redirect(url_for("index"))


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "number_of_classes": len(class_names)
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        UPLOAD_FOLDER,
        exist_ok=True
    )

    print()
    print("========================================")
    print("SERVER STARTING")
    print("========================================")

    print(
        "Model loaded:",
        model is not None
    )

    print(
        "Number of classes:",
        len(class_names)
    )

    print()
    print("Open in browser:")
    print("http://127.0.0.1:5000")
    print("========================================")
    print()

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )