#!/usr/bin/env python3
"""
Fruit Ripeness Comparator -- Agentic Edition
==============================================
Captures a photo of two fruits (Pi Camera), routes each one through
three agents, and reports which is riper.

    Detection Agent      -> what fruit is this?
    Ripeness Agent       -> what % ripe is it? (can exceed 100%)
    Recommendation Agent -> what should I do about it?

Two modes:
    local (default) - free, instant, fully offline. Rule-based/CV heuristics.
    ai              - each agent calls Claude's vision/text API for smarter,
                      more generalizable judgments. Needs ANTHROPIC_API_KEY,
                      internet, and a small per-call cost (a fraction of a
                      cent per fruit with the default Haiku model).

USAGE
    python3 main.py --fruit1 banana --fruit2 mango
    python3 main.py --mode ai
    python3 main.py --mode ai --image test.jpg --fruit1 apple --fruit2 apple

    (Manual --fruit1/--fruit2 skip the Detection Agent for that slot and
    are always more reliable than auto-detection, in either mode.)

Run this from inside the project folder (it imports sibling modules by
plain name, e.g. `segmentation`, `history`).
"""

import argparse
import os
import sys
import time

import cv2

import history
from segmentation import segment_two_fruits
from agents.detection_agent import LocalDetectionAgent, AIDetectionAgent, KNOWN_FRUITS
from agents.ripeness_agent import LocalRipenessAgent, AIRipenessAgent
from agents.recommendation_agent import LocalRecommendationAgent, AIRecommendationAgent


def capture_image(path="capture.jpg", warmup=2):
    """Capture a still photo using the Pi Camera Module via picamera2."""
    from picamera2 import Picamera2

    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    time.sleep(warmup)  # let auto-exposure/white-balance settle
    picam2.capture_file(path)
    picam2.stop()
    print(f"Captured image -> {path}")
    return path


def build_agents(mode, model):
    if mode == "ai":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit(
                "AI mode needs an API key.\n"
                "    export ANTHROPIC_API_KEY=\"sk-ant-...\"\n"
                "Get one at https://console.anthropic.com. Or drop --mode to use "
                "the free local mode instead."
            )
        return AIDetectionAgent(model), AIRipenessAgent(model), AIRecommendationAgent(model)
    return LocalDetectionAgent(), LocalRipenessAgent(), LocalRecommendationAgent()


def main():
    parser = argparse.ArgumentParser(description="Agentic fruit ripeness comparator.")
    parser.add_argument("--mode", choices=["local", "ai"], default="local",
                         help="local = free/offline heuristics, ai = Claude-powered agents")
    parser.add_argument("--fruit1", choices=KNOWN_FRUITS, help="Override auto-detection for the LEFT fruit")
    parser.add_argument("--fruit2", choices=KNOWN_FRUITS, help="Override auto-detection for the RIGHT fruit")
    parser.add_argument("--image", default=None, help="Use an existing photo instead of capturing")
    parser.add_argument("--output", default="result.jpg", help="Where to save the annotated result image")
    parser.add_argument("--model", default="claude-haiku-4-5", help="Claude model to use for --mode ai")
    parser.add_argument("--log", default="history.csv", help="CSV path for history logging")
    parser.add_argument("--no-log", action="store_true", help="Disable history logging")
    args = parser.parse_args()

    detector, ripener, recommender = build_agents(args.mode, args.model)

    image_path = args.image or capture_image()
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Could not read image: {image_path}")

    try:
        fruits = segment_two_fruits(img)
    except ValueError as e:
        sys.exit(f"Segmentation failed: {e}")

    overrides = [args.fruit1, args.fruit2]
    sides = ["Left", "Right"]
    results = []
    for fruit_data, override, side in zip(fruits, overrides, sides):
        if override:
            fruit_type, det_note = override, "manually specified"
        else:
            det = detector.identify(fruit_data)
            fruit_type, det_note = det["fruit_type"], det.get("confidence_note", "")

        ripe = ripener.assess(fruit_data, fruit_type)
        rec_text = recommender.recommend(fruit_type, ripe["ripeness_percent"], ripe.get("cues"))

        results.append({
            "side": side,
            "fruit_type": fruit_type,
            "detection_note": det_note,
            "ripeness_percent": ripe["ripeness_percent"],
            "cues": ripe.get("cues", []),
            "reliability_note": ripe.get("reliability_note", ""),
            "recommendation": rec_text,
            "bbox": fruit_data["bbox"],
        })

    # ---- Report ----
    print(f"\n--- Results ({args.mode} mode) ---")
    for r in results:
        print(f"\n{r['side']} {r['fruit_type']}  (detected via: {r['detection_note']})")
        print(f"  Ripeness: {r['ripeness_percent']:.0f}%")
        if r["reliability_note"]:
            print(f"  Note: {r['reliability_note']}")
        if r["cues"]:
            print(f"  Cues: {', '.join(r['cues'])}")
        print(f"  Recommendation: {r['recommendation']}")

    diff = results[0]["ripeness_percent"] - results[1]["ripeness_percent"]
    print()
    if abs(diff) < 3:
        print(f"Verdict: the {results[0]['side'].lower()} {results[0]['fruit_type']} and the "
              f"{results[1]['side'].lower()} {results[1]['fruit_type']} are about equally ripe.")
    else:
        riper, other = (results[0], results[1]) if diff > 0 else (results[1], results[0])
        print(f"Verdict: the {riper['side'].lower()} {riper['fruit_type']} is riper than the "
              f"{other['side'].lower()} {other['fruit_type']} "
              f"({riper['ripeness_percent']:.0f}% vs {other['ripeness_percent']:.0f}%).")

    # ---- Annotate + save ----
    annotated = img.copy()
    for r in results:
        x, y, w, h = r["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
        label = f"{r['side']} {r['fruit_type']}: {r['ripeness_percent']:.0f}%"
        cv2.putText(annotated, label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite(args.output, annotated)
    print(f"\nAnnotated image saved -> {args.output}")

    # ---- History ----
    if not args.no_log:
        for r in results:
            history.log_result(args.log, r["side"].lower(), r["fruit_type"], r["ripeness_percent"], r["recommendation"])
        print(f"Logged to {args.log}")


if __name__ == "__main__":
    main()
