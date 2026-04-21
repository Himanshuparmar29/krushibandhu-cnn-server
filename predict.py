"""
predict.py — KrushiBandhu Cotton Leaf Disease Prediction
---------------------------------------------------------

Pipeline:
1. Load trained model
2. Load leaf image
3. Leaf segmentation (GrabCut + LAB)
4. Resize + EfficientNet preprocessing
5. Predict disease class
6. RAG knowledge retrieval
7. LLM Crop Doctor advisory
8. Display results

Usage:
    python predict.py
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from crop_doctor import ask_crop_doctor, ask_followup


# ───────────────────────────────────────────
# Default Paths
# ───────────────────────────────────────────

MODEL_PATH       = "model/model_final.keras"
CLASS_NAMES_PATH = "class_names.txt"
OUTPUT_DIR       = "outputs"
IMAGE_SIZE       = 224


# ───────────────────────────────────────────
# User Inputs
# ───────────────────────────────────────────

def get_user_inputs():

    print("\n" + "="*50)
    print("   KrushiBandhu — Cotton Disease Detector")
    print("="*50)

    while True:
        image_path = input("\nEnter path to leaf image: ").strip()
        if os.path.exists(image_path):
            break
        print("File not found. Try again.")

    model_input      = input(f"\nModel path [default: {MODEL_PATH}]: ").strip()
    model_path       = model_input if model_input else MODEL_PATH

    class_input      = input(f"Class names file [default: {CLASS_NAMES_PATH}]: ").strip()
    class_names_path = class_input if class_input else CLASS_NAMES_PATH

    output_input     = input(f"Output folder [default: {OUTPUT_DIR}]: ").strip()
    output_dir       = output_input if output_input else OUTPUT_DIR

    return image_path, model_path, class_names_path, output_dir


# ───────────────────────────────────────────
# Leaf Segmentation
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
# Preprocessing
# ───────────────────────────────────────────

def preprocess_image(image_bgr):

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    segmented_rgb, leaf_mask = segment_leaf(image_bgr)

    resized = cv2.resize(segmented_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    processed = tf.keras.applications.efficientnet.preprocess_input(
        resized.astype(np.float32)
    )

    tensor = np.expand_dims(processed, axis=0)

    return tensor, original_rgb, segmented_rgb, leaf_mask


# ───────────────────────────────────────────
# Prediction
# ───────────────────────────────────────────

def predict(model, image_tensor, class_names):

    probs = model.predict(image_tensor, verbose=0)[0]

    predicted_index = np.argmax(probs)
    predicted_class = class_names[predicted_index]
    confidence      = float(probs[predicted_index])
    scores          = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    return predicted_class, confidence, scores


# ───────────────────────────────────────────
# Visualization
# ───────────────────────────────────────────

def show_results(original_rgb, segmented_rgb, leaf_mask,
                 predicted_class, confidence, scores, output_dir, image_name):

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    fig.suptitle(
        f"Prediction: {predicted_class} | Confidence: {confidence*100:.1f}%",
        fontsize=14, fontweight="bold"
    )

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(leaf_mask, cmap="Greens")
    axes[1].set_title("Leaf Mask")
    axes[1].axis("off")

    axes[2].imshow(segmented_rgb)
    axes[2].set_title("Segmented Leaf")
    axes[2].axis("off")

    classes = list(scores.keys())
    values  = list(scores.values())
    colors  = ["#2ecc71" if c == predicted_class else "#bdc3c7" for c in classes]

    axes[3].barh(classes, values, color=colors)
    axes[3].set_xlim(0, 1)
    axes[3].set_xlabel("Confidence")
    axes[3].set_title("Class Probabilities")
    axes[3].invert_yaxis()

    for i, v in enumerate(values):
        axes[3].text(v + 0.01, i, f"{v*100:.1f}%", va="center")

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"result_{image_name}")
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"\nResult saved to: {save_path}")


# ───────────────────────────────────────────
# Main
# ───────────────────────────────────────────

def main():

    image_path, model_path, class_names_path, output_dir = get_user_inputs()

    # Load class names
    with open(class_names_path) as f:
        class_names = [line.strip() for line in f.readlines()]

    # Load model
    print("\nLoading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")

    # Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError("Could not load image.")

    # Segment + Preprocess
    print("Segmenting leaf...")
    image_tensor, original_rgb, segmented_rgb, leaf_mask = preprocess_image(image_bgr)

    # Predict
    predicted_class, confidence, scores = predict(model, image_tensor, class_names)

    # Print result
    print("\n" + "="*50)
    print("Prediction Result")
    print("="*50)
    print(f"Disease    : {predicted_class}")
    print(f"Confidence : {confidence*100:.2f}%")
    print("\nClass Scores:")
    for cls, score in sorted(scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 20)
        print(f"{cls:<20} {bar:<20} {score*100:.1f}%")

    # Visualize
    image_name = os.path.basename(image_path)
    show_results(
        original_rgb, segmented_rgb, leaf_mask,
        predicted_class, confidence, scores,
        output_dir, image_name
    )

    # Language selection
    print("\nSelect response language:")
    print("  1. English  2. Hindi  3. Gujarati")
    lang_choice = input("Enter choice (1/2/3) [default: 1]: ").strip()
    language    = {"1": "english", "2": "hindi", "3": "gujarati"}.get(lang_choice, "english")

    # Crop Doctor advisory
    print("\nConsulting Crop Doctor...")
    advisory = ask_crop_doctor(predicted_class, confidence, language)

    print("\n" + "="*50)
    print("KrushiBandhu Crop Doctor Advisory")
    print("="*50)
    print(advisory)
    print("="*50)

    # Save advisory
    advisory_path = os.path.join(output_dir, f"advisory_{os.path.splitext(image_name)[0]}.txt")
    with open(advisory_path, "w", encoding="utf-8") as f:
        f.write(f"Disease    : {predicted_class}\n")
        f.write(f"Confidence : {confidence*100:.2f}%\n")
        f.write(f"Language   : {language.capitalize()}\n\n")
        f.write(advisory)
    print(f"Advisory saved to: {advisory_path}")

    # Follow-up Q&A
    print("\nDo you have any questions? (y/n): ", end="")
    while input().strip().lower() == "y":
        question = input("Your question: ").strip()
        if question:
            answer = ask_followup(question, predicted_class, language)
            print("\n" + "-"*50)
            print(answer)
            print("-"*50)
        print("\nAnother question? (y/n): ", end="")

    # Loop
    again = input("\nAnalyse another leaf? (y/n): ").strip().lower()
    if again == "y":
        main()
    else:
        print("\nThank you for using KrushiBandhu!\n")


if __name__ == "__main__":
    main()
