"""
Segmentation
------------
Finds the two fruit regions in a photo using classical computer vision.

This step deliberately does NOT use an LLM: "find two high-saturation
blobs against a plain background" is a solved, bounded problem, so a
model call here would only add cost and latency without improving
results. The three agents (detection, ripeness, recommendation) are
where the harder, fuzzier reasoning actually happens.
"""

import base64

import cv2
import numpy as np


def segment_two_fruits(bgr_image):
    """
    Locate the two largest fruit-sized blobs in the image.

    Returns a list of 2 dicts, sorted left-to-right (so [0] = fruit1,
    [1] = fruit2), each with:
        bbox, crop, mask, area_px, aspect_ratio, mean_hue, mean_sat, mean_val

    Raises ValueError if it can't confidently find two regions.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    # Plain backgrounds (white/grey/dark cloth or paper) tend to have low
    # saturation; fruit skin tends to be much more saturated. Otsu finds
    # a good split point automatically.
    blurred = cv2.GaussianBlur(sat, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 2000]
    contours.sort(key=cv2.contourArea, reverse=True)

    if len(contours) < 2:
        raise ValueError(
            f"Only found {len(contours)} fruit-sized region(s). "
            "Check lighting/background contrast, or try a plainer backdrop."
        )

    results = []
    for c in contours[:2]:
        x, y, w, h = cv2.boundingRect(c)

        mask = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        roi_mask = mask[y:y + h, x:x + w]
        roi = bgr_image[y:y + h, x:x + w]
        mask_bool = roi_mask > 0

        roi_hsv = hsv[y:y + h, x:x + w]
        hue_px = roi_hsv[:, :, 0][mask_bool].astype(np.float64)
        sat_px = roi_hsv[:, :, 1][mask_bool].astype(np.float64)
        mean_sat = float(sat_px.mean())
        mean_val = float(roi_hsv[:, :, 2][mask_bool].mean())

        # Hue is circular -- 0 and 179 are both "red" in OpenCV's 0-179
        # scale. A plain arithmetic mean is wrong here: a strawberry with
        # pixels split between hue~3 and hue~176 (both visually red) would
        # naively average to ~90 (green), which is exactly backwards. Use a
        # proper circular mean instead, weighted by saturation so washed-out
        # or shadowed pixels don't skew the result.
        weights = np.clip(sat_px, 1, None)
        hue_rad = hue_px * (np.pi / 90.0)  # OpenCV hue (0-179) -> radians (0-2pi)
        mean_sin = np.average(np.sin(hue_rad), weights=weights)
        mean_cos = np.average(np.cos(hue_rad), weights=weights)
        mean_hue = float((np.degrees(np.arctan2(mean_sin, mean_cos)) % 360) / 2)

        (_cx, _cy), (rw, rh), _angle = cv2.minAreaRect(c)
        long_side, short_side = max(rw, rh), max(min(rw, rh), 1e-3)

        results.append({
            "bbox": (x, y, w, h),
            "crop": roi,
            "mask": roi_mask,
            "area_px": float(cv2.contourArea(c)),
            "aspect_ratio": float(long_side / short_side),
            "mean_hue": mean_hue,
            "mean_sat": mean_sat,
            "mean_val": mean_val,
        })

    results.sort(key=lambda r: r["bbox"][0])  # left-to-right
    return results


def encode_image_b64(bgr_image, max_dim=768, quality=85):
    """
    Encode an OpenCV BGR image to base64 JPEG for the Claude API.
    Downscales first -- ripeness cues don't need full resolution, and a
    smaller image means fewer tokens (cheaper, faster) per API call.
    """
    h, w = bgr_image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        bgr_image = cv2.resize(bgr_image, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode image")
    return base64.b64encode(buf).decode("utf-8")
