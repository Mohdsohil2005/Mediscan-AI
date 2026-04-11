import cv2
import numpy as np


MODEL_IMAGE_SIZE = (150, 150)
MIN_CONFIDENCE = 0.68
MIN_MARGIN = 0.18


def normalize_prediction_label(raw_label):
    normalized = (raw_label or "").strip().lower()
    if "covid" in normalized:
        return "COVID"
    if "normal" in normalized:
        return "NORMAL"
    if "pneumonia" in normalized:
        return "PNEUMONIA"
    return (raw_label or "UNKNOWN").strip().upper()


def _resize_and_stack(gray_image):
    resized = cv2.resize(gray_image, MODEL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return np.stack([normalized, normalized, normalized], axis=-1)


def _crop_focus_region(gray_image):
    height, width = gray_image.shape[:2]
    top = int(height * 0.06)
    bottom = int(height * 0.94)
    left = int(width * 0.1)
    right = int(width * 0.9)
    cropped = gray_image[top:bottom, left:right]
    return cropped if cropped.size else gray_image


def extract_focus_roi(image_bgr):
    height, width = image_bgr.shape[:2]
    image_area = float(max(height * width, 1))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    edges = cv2.Canny(equalized, 45, 140)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = None
    best_score = -1.0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < image_area * 0.18:
            continue

        aspect_ratio = w / float(max(h, 1))
        if aspect_ratio < 0.5 or aspect_ratio > 1.65:
            continue

        coverage = area / image_area
        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)
        center_offset = abs((center_x / width) - 0.5) + abs((center_y / height) - 0.5)
        score = coverage - (center_offset * 0.35)

        if score > best_score:
            best_score = score
            best_box = (x, y, w, h)

    if best_box:
        x, y, w, h = best_box
        pad_x = int(w * 0.04)
        pad_y = int(h * 0.04)
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, width)
        y2 = min(y + h + pad_y, height)
        roi = image_bgr[y1:y2, x1:x2]
        if roi.size:
            return roi, {"detected": True, "box": (x1, y1, x2, y2)}

    fallback_gray = _crop_focus_region(gray)
    fallback = cv2.cvtColor(fallback_gray, cv2.COLOR_GRAY2BGR)
    return fallback, {"detected": False, "box": None}


def build_tta_batch(image_bgr):
    roi_bgr, _ = extract_focus_roi(image_bgr)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    focused = _crop_focus_region(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    focused_enhanced = clahe.apply(focused)

    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    focused_normalized = cv2.normalize(focused, None, 0, 255, cv2.NORM_MINMAX)
    sharper = cv2.addWeighted(enhanced, 1.15, normalized, -0.15, 0)
    focused_sharper = cv2.addWeighted(focused_enhanced, 1.18, focused_normalized, -0.18, 0)
    denoised = cv2.bilateralFilter(enhanced, 5, 35, 35)

    variants = [
        _resize_and_stack(gray),
        _resize_and_stack(enhanced),
        _resize_and_stack(sharper),
        _resize_and_stack(focused),
        _resize_and_stack(focused_enhanced),
        _resize_and_stack(focused_sharper),
        _resize_and_stack(denoised),
    ]
    return np.asarray(variants, dtype="float32")


def summarize_prediction(predictions, labels):
    weights = np.array([1.0, 1.1, 1.15, 1.05, 1.2, 1.2, 1.1], dtype="float32")
    if len(predictions) == len(weights):
        mean_scores = np.average(predictions, axis=0, weights=weights)
    else:
        mean_scores = np.mean(predictions, axis=0)
    top_index = int(np.argmax(mean_scores))
    sorted_scores = np.sort(mean_scores)[::-1]
    confidence = float(mean_scores[top_index])
    margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else confidence
    predicted_label = normalize_prediction_label(labels[top_index])
    is_uncertain = confidence < MIN_CONFIDENCE or margin < MIN_MARGIN
    return {
        "predicted_label": predicted_label,
        "confidence_score": confidence,
        "margin": margin,
        "is_uncertain": is_uncertain,
        "mean_scores": mean_scores.tolist(),
    }
