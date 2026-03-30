# Jarvis - Clap-Activated Browser Automation

Jarvis is a Python script that listens to your microphone, detects clap patterns in real time, and triggers browser actions.

## Clap Actions

| Clap pattern | Action |
|---|---|
| Single clap | Open ChatGPT (`https://chatgpt.com/`) |
| Double clap | Open YouTube (`https://www.youtube.com`) |
| Triple clap | Open browser homepage (`about:blank`) |

Action mapping is defined in the `ACTIONS` dictionary inside `jarvis.py`.

## Detection Overview

For each 25 ms audio chunk, Jarvis extracts acoustic features and scores clap confidence.

Pipeline:

1. Energy gate (RMS above calibrated threshold)
2. Feature extraction (crest factor, spectral flatness, centroid, band energy ratio, zero-crossing rate, kurtosis)
3. Confidence scoring (weighted multi-feature score)
4. Pattern grouping inside a clap window
5. Action dispatch with cooldown

The RMS threshold is auto-calibrated and periodically adapted, so there is no fixed hardcoded threshold constant.

## Requirements

- Python 3.10+
- Microphone input available to the system
- Packages listed in `requirements.txt`:
	- `sounddevice`
	- `numpy`
	- `scipy`

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python jarvis.py
```

On startup, Jarvis:

1. Samples ambient audio for calibration (`CALIBRATION_SECONDS`)
2. Computes a fused spike factor (Neyman-Pearson + empirical SNR + percentile method)
3. Prints a calibration report and starts listening

Press `Ctrl+C` to stop.

## Main Configuration (jarvis.py)

These are the key tunables in the script:

- `SAMPLE_RATE = 44100`
- `CHUNK_DURATION = 0.025`
- `CALIBRATION_SECONDS = 3.0`
- `CLAP_WINDOW = 0.75`
- `MIN_CLAP_GAP = 0.10`
- `ACTION_COOLDOWN = 2.0`
- `ADAPT_INTERVAL = 10.0`
- `MIN_CONFIDENCE = 0.20`
- `CLAP_BAND_LO = 1000`
- `CLAP_BAND_HI = 6000`
- `TARGET_PFA = 1e-4`

## Troubleshooting

| Issue | What to do |
|---|---|
| `PortAudioError` | Check that a microphone is connected and OS mic permission is enabled. |
| Claps not detected | Decrease `MIN_CONFIDENCE` slightly or increase `CALIBRATION_SECONDS` for better baseline estimation. |
| Too many false detections | Increase `MIN_CONFIDENCE`, increase `MIN_CLAP_GAP`, or reduce room noise during calibration. |
| Wrong site opens | Update URL/action entries in `ACTIONS` in `jarvis.py`. |

## Notes

- If more than 3 claps are detected in a window, no browser action is mapped by default.
- If actions fire too quickly, `ACTION_COOLDOWN` prevents immediate retriggering.
