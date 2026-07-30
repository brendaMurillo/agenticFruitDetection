#!/usr/bin/env python3

import os
import subprocess
import time
from datetime import datetime

from gpiozero import Button, LED

# Physical pin 11 = GPIO 17
button = Button(17, pull_up=True, bounce_time=0.2)

# Physical pin 13 = GPIO 27
led = LED(27)

PHOTO_DIR = os.path.expanduser("~/photos")
os.makedirs(PHOTO_DIR, exist_ok=True)

print("Ready. Press and release the button. Ctrl+C to quit.")

try:
    while True:
        button.wait_for_press()
        led.on()
        print("Button pressed. Release it...", flush=True)

        button.wait_for_release()
        print("Taking photo...", flush=True)

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        filename = os.path.join(PHOTO_DIR, f"photo_{stamp}.jpg")

        try:
            subprocess.run(
                [
                    "rpicam-jpeg",
                    "--output", filename,
                    "--timeout", "3000",
                    "--nopreview",
                ],
                check=True,
            )

            print(f"Saved {filename}", flush=True)

        finally:
            led.off()

        time.sleep(0.3)

except KeyboardInterrupt:
    print("\nStopping.")

finally:
    led.off()
