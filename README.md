# Fruit Ripeness Comparator (Agentic Edition)

Takes a photo of two fruits side by side, runs each one through three
agents, and tells you which is riper — plus what to actually do about it.

```
Photo --> Segmentation (classical CV) --> for each fruit:
              Detection Agent      "what fruit is this?"
              Ripeness Agent       "what % ripe? (0-150%)"
              Recommendation Agent "what should I do?"
          --> Verdict + annotated photo + history log
```

## Project layout

```
main.py                          orchestrator / CLI entry point
segmentation.py                  finds the two fruit blobs in the photo (classical CV)
history.py                       logs every scan to a CSV
agents/
  detection_agent.py             identifies the fruit type
  ripeness_agent.py               scores ripeness 0-150%
  recommendation_agent.py         turns that into advice
```

Each agent has two interchangeable implementations:

| | Local (default) | AI (`--mode ai`) |
|---|---|---|
| Cost | Free | ~fraction of a cent/fruit (Haiku 4.5 pricing) |
| Needs internet | No | Yes |
| How it decides | Hand-tuned color/shape rules | Claude vision API |
| Best for | Quick, offline, distinctly-different fruits | Higher accuracy, especially for visually similar round fruits (apple/tomato/orange) |

## Setup

```bash
cd fruit_ripeness_agents
pip3 install -r requirements.txt --break-system-packages
```

For AI mode, get a key at https://console.anthropic.com and:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```
(Add that line to `~/.bashrc` so you don't have to re-set it every session.)

## Physical setup

Place fruit1 on the **left** half of the frame, fruit2 on the **right**,
against a plain, high-contrast background (white poster board or dark
cloth). Even lighting, no strong shadows or glare.

## Usage

```bash
# Free, offline, auto-detect both fruits
python3 main.py

# Tell it what the fruits are (always more reliable than auto-detect)
python3 main.py --fruit1 banana --fruit2 mango

# AI mode -- better auto-detection and more nuanced ripeness/advice
python3 main.py --mode ai

# Test against an existing photo instead of capturing a new one
python3 main.py --image test.jpg --fruit1 apple --fruit2 apple
```

Output: a printed report, an annotated `result.jpg` with boxes and %
labels, and a row appended to `history.csv` for each fruit.

## Honesty about accuracy

- **Reliable ripeness scoring:** banana, mango, strawberry, tomato,
  avocado, apple (color visibly changes as these ripen).
- **Low-confidence ripeness scoring:** orange, watermelon (exterior
  color barely changes with ripeness — the script flags this rather
  than pretending otherwise). For watermelon specifically, tap-sound
  analysis would be a much better signal than vision if you want to
  extend this later.
- **Auto-detection (local mode):** works well for shapes that stand
  out (banana, mango, avocado, watermelon) but can genuinely mix up
  round fruits of similar color (apple vs. tomato vs. orange), since
  it's just hue + size + aspect ratio rules. AI mode is meaningfully
  more accurate here. Either way, `--fruit1`/`--fruit2` overrides are
  always the most reliable option.
- **Recommendations respect fruit biology:** banana/mango/avocado/tomato/apple
  keep ripening after picking (climacteric), so "wait 2 days" is real
  advice. Strawberries, oranges, and watermelon do **not** ripen further
  once picked — the script says so instead of telling you to wait for
  something that won't happen.

## Calibrating for your setup

The HSV thresholds in `agents/ripeness_agent.py` and the shape
thresholds in `agents/detection_agent.py` are reasonable starting
points, not universal constants — your camera, lighting, and distance
from the fruit will shift the numbers. Take a few reference photos of
known-unripe and known-ripe specimens, and adjust the ranges to match.

## Web UI

Instead of reading terminal output, you can view the photo and results in
a browser. There are two ways to run it -- **running it on the Pi is
recommended**, since it removes the manual `scp` step entirely.

### Option A: Run it on the Pi (no more copying photos over)

Since `button_camera.py` already saves photos locally on the Pi, running
the web app there too means it can see new photos immediately -- no
transfer step. This is a one-time setup:

```bash
# From your Mac, copy the whole project to the Pi (one time)
scp -r ~/Downloads/fruit_ripeness_agents piuser@raspberrypi.local:~/

# SSH in and install dependencies (one time; opencv can take a few
# minutes to install over a hotspot connection)
ssh piuser@raspberrypi.local
cd fruit_ripeness_agents
pip3 install -r requirements.txt --break-system-packages
```

Then, **every time you want to use it**, open two separate terminal
windows/tabs on your Mac, each SSH'd into the Pi (`ssh
piuser@raspberrypi.local` in both):

**Terminal 1** -- capture, same as always:
```bash
python3 button_camera.py
```

**Terminal 2** -- the web app, pointed at the Pi's own photos folder:
```bash
cd fruit_ripeness_agents
PHOTOS_DIR=/home/piuser/photos python3 webapp.py
```

It'll print something like:
```
http://<this-device's-IP>:5001   (from another device on the same network)
```

Find the Pi's actual IP with `hostname -I` if you don't already have it,
then open `http://<that-ip>:5001` **in your Mac's browser**. Now the
loop is: press the button → click "refresh" in the browser → pick
fruits → click "Analyze". No `scp`, no filename-copying.

### Option B: Run it on your Mac (original workflow)

```bash
python3 webapp.py
```

Then open **http://127.0.0.1:5001**. This still needs the manual `scp`
step from before -- automatically shows the newest photo in
`~/Downloads/RaspberryPiPhotos` (override with `PHOTOS_DIR=...`).

### Either way

Lets you pick fruit types (or leave on auto-detect) and switch modes,
and displays the annotated photo, a ripeness gauge per fruit, and the
recommendation -- using the exact same agents as `main.py`. Every scan
still gets logged to the same `history.csv`.

## Natural next steps

- **Track a single fruit's ripening curve:** photograph the same fruit
  in the same spot daily; `history.csv` becomes a real time series you
  could plot with matplotlib.
- **Better auto-detection:** train a small image classifier (e.g.
  transfer learning on MobileNet with the public Fruits-360 dataset)
  and swap it in as a new `DetectionAgent` implementation — the
  interface (`identify(fruit_data) -> {"fruit_type", "confidence_note"}`)
  is designed to make that a drop-in replacement.
- **Watermelon-specific sensing:** the Pi's mic + a tap-sound
  frequency analysis is a legitimately better ripeness signal than
  anything a photo can give you.
