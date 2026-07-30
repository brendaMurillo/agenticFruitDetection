"""
History Logger
---------------
Optional extra: appends every scan to a local CSV so you can track
ripening over time -- e.g. photograph the same fruit daily and watch
the percentage climb -- or just keep a record of past comparisons.

Not an "agent" in the reasoning sense, just bookkeeping, but useful
enough to include by default.
"""

import csv
import os
from datetime import datetime

FIELDNAMES = ["timestamp", "slot", "fruit_type", "ripeness_percent", "recommendation"]


def log_result(csv_path, slot, fruit_type, ripeness_percent, recommendation):
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "slot": slot,
            "fruit_type": fruit_type,
            "ripeness_percent": f"{ripeness_percent:.1f}",
            "recommendation": recommendation,
        })


def read_recent(csv_path, fruit_type=None, limit=5):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if fruit_type:
        rows = [r for r in rows if r["fruit_type"] == fruit_type]
    return rows[-limit:]
