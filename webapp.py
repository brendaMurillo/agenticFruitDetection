#!/usr/bin/env python3
"""
Web UI
------
A small local Flask app that shows the latest captured photo alongside
the ripeness results -- same three-agent pipeline as main.py, with a
browser front end instead of the terminal.

Run:
    python3 webapp.py
Then open http://127.0.0.1:5001 in your browser.

By default it looks for photos in ~/Downloads/RaspberryPiPhotos (the
folder you've been scp'ing photos into). Override with:
    PHOTOS_DIR=/some/other/path python3 webapp.py
"""

import glob
import os

import cv2
from flask import Flask, render_template, request, send_from_directory

import history
from agents.detection_agent import KNOWN_FRUITS, AIDetectionAgent, LocalDetectionAgent
from agents.recommendation_agent import AIRecommendationAgent, LocalRecommendationAgent
from agents.ripeness_agent import AIRipenessAgent, LocalRipenessAgent
from segmentation import segment_two_fruits

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.environ.get("PHOTOS_DIR", os.path.expanduser("~/Downloads/RaspberryPiPhotos"))
RESULTS_DIR = os.path.join(PROJECT_DIR, "webapp_results")
HISTORY_CSV = os.path.join(PROJECT_DIR, "history.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__)

# Tier boundaries mirror agents/recommendation_agent.py exactly -- this is
# also what the gauge's colored zones are drawn from, so the UI always
# matches the actual logic instead of drifting out of sync with it.
TIERS = [
    (40, "UNRIPE", "#7C9A5A"),
    (85, "RIPENING", "#C4A83B"),
    (108, "PEAK", "#E8B93F"),
    (130, "PAST PEAK", "#D2762E"),
    (150, "OVERRIPE", "#8B3A2E"),
]


def tier_info(pct):
    for threshold, label, color in TIERS:
        if pct <= threshold:
            return label, color
    return TIERS[-1][1], TIERS[-1][2]


def latest_photo():
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(PHOTOS_DIR, p)))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def build_agents(mode):
    if mode == "ai":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                'AI mode needs an API key. Quit this app, run '
                'export ANTHROPIC_API_KEY="sk-ant-..." in the terminal, then restart it.'
            )
        return AIDetectionAgent(), AIRipenessAgent(), AIRecommendationAgent()
    return LocalDetectionAgent(), LocalRipenessAgent(), LocalRecommendationAgent()


def render(mode="local", fruit1="", fruit2="", result=None, error=None):
    photo_path = latest_photo()
    photo_name = os.path.basename(photo_path) if photo_path else None
    return render_template(
        "index.html",
        known_fruits=KNOWN_FRUITS,
        photos_dir=PHOTOS_DIR,
        photo_name=photo_name,
        result=result,
        error=error,
        mode=mode,
        fruit1=fruit1,
        fruit2=fruit2,
    )


@app.route("/", methods=["GET"])
def index():
    return render()


@app.route("/analyze", methods=["POST"])
def analyze():
    mode = request.form.get("mode", "local")
    fruit1 = request.form.get("fruit1") or ""
    fruit2 = request.form.get("fruit2") or ""

    photo_path = latest_photo()
    if not photo_path:
        return render(mode, fruit1, fruit2, error=f"No photos found in {PHOTOS_DIR}. Take one, scp it over, then reload.")

    try:
        detector, ripener, recommender = build_agents(mode)
    except RuntimeError as e:
        return render(mode, fruit1, fruit2, error=str(e))

    img = cv2.imread(photo_path)
    if img is None:
        return render(mode, fruit1, fruit2, error="Could not read that photo file -- it may be corrupted or still copying.")

    try:
        fruits = segment_two_fruits(img)
    except ValueError as e:
        return render(mode, fruit1, fruit2, error=str(e))

    overrides = [fruit1 or None, fruit2 or None]
    sides = ["Left", "Right"]
    items = []
    for fruit_data, override, side in zip(fruits, overrides, sides):
        if override:
            fruit_type, det_note = override, "manually specified"
        else:
            det = detector.identify(fruit_data)
            fruit_type, det_note = det["fruit_type"], det.get("confidence_note", "")

        ripe = ripener.assess(fruit_data, fruit_type)
        pct = ripe["ripeness_percent"]
        rec_text = recommender.recommend(fruit_type, pct, ripe.get("cues"))
        tier_label, tier_color = tier_info(pct)

        items.append({
            "side": side,
            "fruit_type": fruit_type,
            "detection_note": det_note,
            "ripeness_percent": pct,
            "gauge_pct": min(pct, 150) / 150 * 100,
            "tier_label": tier_label,
            "tier_color": tier_color,
            "cues": ripe.get("cues", []),
            "reliability_note": ripe.get("reliability_note", ""),
            "recommendation": rec_text,
            "bbox": fruit_data["bbox"],
        })

    # Annotate + save a result image
    annotated = img.copy()
    for r in items:
        x, y, w, h = r["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 4)
        label = f"{r['side']} {r['fruit_type']}: {r['ripeness_percent']:.0f}%"
        cv2.putText(annotated, label, (x, max(y - 12, 24)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 0), 3)
    photo_name = os.path.basename(photo_path)
    result_filename = f"result_{os.path.splitext(photo_name)[0]}.jpg"
    cv2.imwrite(os.path.join(RESULTS_DIR, result_filename), annotated)

    for r in items:
        history.log_result(HISTORY_CSV, r["side"].lower(), r["fruit_type"], r["ripeness_percent"], r["recommendation"])

    diff = items[0]["ripeness_percent"] - items[1]["ripeness_percent"]
    if abs(diff) < 3:
        verdict = f"The {items[0]['side'].lower()} {items[0]['fruit_type']} and the {items[1]['side'].lower()} {items[1]['fruit_type']} are about equally ripe."
    else:
        riper, other = (items[0], items[1]) if diff > 0 else (items[1], items[0])
        verdict = f"The {riper['side'].lower()} {riper['fruit_type']} is riper than the {other['side'].lower()} {other['fruit_type']}."

    result = {"fruits": items, "verdict": verdict, "result_image": result_filename, "mode": mode}
    return render(mode, fruit1, fruit2, result=result)


@app.route("/photos/<path:filename>")
def serve_photo(filename):
    return send_from_directory(PHOTOS_DIR, filename)


@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    print(f"Watching for photos in: {PHOTOS_DIR}")
    print("Open this in a browser:")
    print("  http://127.0.0.1:5001            (if you're on this machine)")
    print("  http://<this-device's-IP>:5001   (from another device on the same network)")
    app.run(host="0.0.0.0", port=5001, debug=False)
