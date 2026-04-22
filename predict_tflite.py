"""
predict_tflite.py — TFLite inference for KrushiBandhu
─────────────────────────────────────────────────────
Drop-in replacement for the prediction pipeline that uses
tflite-runtime (~30 MB) instead of full TensorFlow (~1.7 GB).

Only the server uses this module.  The original predict.py
is kept for local experimentation with full TensorFlow.
"""

import numpy as np
import cv2


IMAGE_SIZE = 224


# ───────────────────────────────────────────
# Leaf Segmentation  (identical to predict.py — pure OpenCV)
# ───────────────────────────────────────────

def segment_leaf(image_bgr):
    """
    Segments the leaf from background using GrabCut + LAB color space.
    Handles diseased leaves (brown, yellow, spotted) reliably.
    Returns segmented RGB image and binary mask.
    """
    h, w = image_bgr.shape[:2]

    kernel_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    # Step 1 — GrabCut with rectangle
    margin_x = max(5, int(w * 0.05))
    margin_y = max(5, int(h * 0.05))
    rect      = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)

    gc_mask   = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image_bgr, gc_mask, rect, bgd_model, fgd_model,
                    iterCount=8, mode=cv2.GC_INIT_WITH_RECT)
        grabcut_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    except cv2.error:
        grabcut_mask = np.zeros((h, w), np.uint8)
        grabcut_mask[margin_y:h-margin_y, margin_x:w-margin_x] = 255

    # Step 2 — LAB 'a' channel (detects all plant tissue regardless of color)
    lab      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    a_inv    = cv2.bitwise_not(lab[:, :, 1])
    _, lab_mask = cv2.threshold(a_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lab_mask    = cv2.morphologyEx(lab_mask, cv2.MORPH_CLOSE,  kernel_close)
    lab_mask    = cv2.morphologyEx(lab_mask, cv2.MORPH_DILATE, kernel_dilate)

    # Step 3 — Combine both masks
    combined = cv2.bitwise_or(grabcut_mask, lab_mask)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

    # Step 4 — Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    if num_labels > 1:
        largest    = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    else:
        final_mask = combined

    # Step 5 — Fill internal holes
    flood = final_mask.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    final_mask = cv2.bitwise_or(final_mask, cv2.bitwise_not(flood))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)

    # Replace background with white
    image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_3ch   = cv2.merge([final_mask, final_mask, final_mask])
    background = np.ones_like(image_rgb) * 255
    segmented  = np.where(mask_3ch > 0, image_rgb, background).astype(np.uint8)

    return segmented, final_mask


# ───────────────────────────────────────────
# Preprocessing (EfficientNet-compatible, no TF dependency)
# ───────────────────────────────────────────

def _efficientnet_preprocess(img_array):
    """
    Replicate tf.keras.applications.efficientnet.preprocess_input
    using pure NumPy.  EfficientNet uses 'torch' mode:
        x = x / 255.0
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    """
    x = img_array.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return x


def preprocess_image(image_bgr):
    """Returns (tensor, original_rgb, segmented_rgb, leaf_mask)."""
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    segmented_rgb, leaf_mask = segment_leaf(image_bgr)

    resized = cv2.resize(segmented_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    processed = _efficientnet_preprocess(resized)

    tensor = np.expand_dims(processed, axis=0)   # shape (1, 224, 224, 3)

    return tensor, original_rgb, segmented_rgb, leaf_mask


# ───────────────────────────────────────────
# Prediction (TFLite)
# ───────────────────────────────────────────

def predict(interpreter, input_details, output_details, image_tensor, class_names):
    """
    Run TFLite inference.

    Returns (predicted_class, confidence, scores_dict).
    """
    # TFLite expects float32; the float16 quantised model still accepts f32 input
    image_tensor = image_tensor.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], image_tensor)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]

    # Apply softmax if model outputs logits (safety check)
    if probs.min() < 0 or probs.sum() < 0.99 or probs.sum() > 1.01:
        exp_probs = np.exp(probs - np.max(probs))
        probs = exp_probs / exp_probs.sum()

    predicted_index = int(np.argmax(probs))
    predicted_class = class_names[predicted_index]
    confidence      = float(probs[predicted_index])
    scores          = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    return predicted_class, confidence, scores
