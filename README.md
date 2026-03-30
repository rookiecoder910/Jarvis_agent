# 🎤 Jarvis — Clap-Activated Browser Automation

A Python script that listens to your microphone and detects clap patterns to automate browser actions.

| Pattern | Action |
|---|---|
| **Single clap** | Opens [ChatGPT](https://chat.openai.com) |
| **Double clap** | Opens [YouTube](https://www.youtube.com) |
| **Triple clap** | Opens browser homepage |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Jarvis
python jarvis.py

# 3. Clap near your mic! Press Ctrl+C to stop.
```

## Tuning Sensitivity

Open `jarvis.py` and adjust these constants at the top:

| Constant | Default | Description |
|---|---|---|
| `THRESHOLD` | `0.4` | Amplitude spike needed to register a clap (0.0–1.0). **Raise** if too many false positives; **lower** if claps aren't picked up. |
| `CLAP_WINDOW` | `1.5` s | Time window to group claps into a pattern. |
| `MIN_CLAP_INTERVAL` | `0.08` s | Debounce gap — ignores echo/re-triggers within this time. |
| `ACTION_COOLDOWN` | `2.0` s | Cooldown after a browser action fires. |

## Troubleshooting

| Issue | Fix |
|---|---|
| `PortAudioError` | Make sure a microphone is connected and your OS has granted mic permissions. |
| Nothing detected | Lower `THRESHOLD` (e.g. `0.2`). |
| False positives | Raise `THRESHOLD` (e.g. `0.6`), or increase `MIN_CLAP_INTERVAL`. |
| Wrong URLs | Edit the `ACTIONS` dictionary in `jarvis.py`. |

## Requirements

- Python 3.10+
- A working microphone
- Windows (primary), macOS / Linux also supported
