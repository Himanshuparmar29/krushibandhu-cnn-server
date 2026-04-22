"""
api_server.py — KrushiBandhu Disease Detection Backend (TFLite)
───────────────────────────────────────────────────────────────
Handles:
  • POST /predict   — leaf disease prediction (TFLite model)
  • POST /followup  — farmer follow-up question (LLM)
  • GET  /health    — health check

Uses tflite-runtime (~30 MB RAM) instead of full TensorFlow (~1.7 GB)
so it can run on Render free tier (512 MB RAM).

Auth is handled by Supabase (client-side). This server is ML-only.

Run locally:
  pip install -r requirements.txt
  python api_server.py

Deploy (Render.com / Railway):
  gunicorn api_server:app --timeout 120 --workers 1
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64, tempfile, os, traceback, re
import numpy as np
import cv2

# ── Import prediction pipeline ────────────────────────────────────────────────
from predict_tflite import preprocess_image, predict as run_predict
from crop_doctor import ask_crop_doctor

app = Flask(__name__)
CORS(app)

# ── Load TFLite model at startup ──────────────────────────────────────────────
TFLITE_PATH = "model/model_final.tflite"
CLASS_NAMES_PATH = "class_names.txt"

print("Loading TFLite model...")
try:
    # Try tflite-runtime first (deployed), fall back to tf.lite (local dev)
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

    INTERPRETER = tflite.Interpreter(model_path=TFLITE_PATH)
    INTERPRETER.allocate_tensors()
    INPUT_DETAILS  = INTERPRETER.get_input_details()
    OUTPUT_DETAILS = INTERPRETER.get_output_details()
    NAMES = open(CLASS_NAMES_PATH).read().splitlines()
    print(f"Model loaded -- {len(NAMES)} classes: {NAMES}")
except Exception as e:
    print(f"Model load failed: {e}")
    INTERPRETER = None
    INPUT_DETAILS = None
    OUTPUT_DETAILS = None
    NAMES = []

# ── Advisory text parser ───────────────────────────────────────────────────────
_SECTION_PATTERNS = [
    ("what_is_happening",     r"WHAT IS HAPPENING"),
    ("why_it_happened",       r"WHY THIS HAPPENED"),
    ("immediate_action",      r"IMMEDIATE ACTION"),
    ("organic_remedy",        r"ORGANIC\s*/?\s*LOW[- ]COST REMEDY|ORGANIC REMEDY"),
    ("prevention",            r"HOW TO PREVENT|PREVENTION"),
    ("when_to_see_specialist",r"WHEN TO SEE A SPECIALIST|SPECIALIST"),
]

def _parse_advisory(text: str) -> dict:
    result = {key: "" for key, _ in _SECTION_PATTERNS}
    result["raw_advisory"] = text.strip()

    found = []
    for key, pattern in _SECTION_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            found.append((m.start(), m.end(), key))

    if not found:
        result["what_is_happening"] = text.strip()
        return result

    found.sort(key=lambda x: x[0])
    for i, (start, end, key) in enumerate(found):
        next_start = found[i + 1][0] if i + 1 < len(found) else len(text)
        content = text[end:next_start].strip(" :\n\r\t—–-")
        result[key] = content.strip()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE: POST /predict
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict_route():
    if INTERPRETER is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data    = request.json
        img_b64 = data.get("image_base64", "")

        if not img_b64:
            return jsonify({"error": "No image provided"}), 400

        # Decode image
        img_bytes = base64.b64decode(img_b64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(img_bytes)
            path = f.name

        bgr = cv2.imread(path)
        os.remove(path)

        if bgr is None:
            return jsonify({"error": "Could not decode image"}), 400

        language = (data.get("language", "english") or "english").strip().lower()

        # Run ML pipeline
        tensor, _, seg, mask = preprocess_image(bgr)
        pred_class, conf, scores = run_predict(
            INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS, tensor, NAMES
        )

        # Get advisory from LLM
        advisory_text = ask_crop_doctor(pred_class, conf, language)
        sections = _parse_advisory(advisory_text)

        return jsonify({
            "disease":                pred_class,
            "confidence":             conf,
            "scores":                 scores,
            "what_is_happening":      sections["what_is_happening"],
            "why_it_happened":        sections["why_it_happened"],
            "immediate_action":       sections["immediate_action"],
            "organic_remedy":         sections["organic_remedy"],
            "prevention":             sections["prevention"],
            "when_to_see_specialist": sections["when_to_see_specialist"],
            "advisory":               sections["raw_advisory"],
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE: POST /followup
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/followup", methods=["POST"])
def followup_route():
    try:
        from crop_doctor import ask_followup
        data         = request.json
        question     = (data.get("question", "") or "").strip()
        disease_name = (data.get("disease_name", "") or "").strip()
        language     = (data.get("language", "english") or "english").strip().lower()

        if not question:
            return jsonify({"error": "No question provided"}), 400

        answer = ask_followup(question, disease_name, language)
        return jsonify({"answer": answer}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE: GET /health
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok" if INTERPRETER is not None else "degraded",
        "model":   "loaded" if INTERPRETER is not None else "not loaded",
        "classes": NAMES,
    })


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  KrushiBandhu Disease Detection API (TFLite)")
    print("="*50)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
