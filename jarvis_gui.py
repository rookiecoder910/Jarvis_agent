"""
Jarvis GUI — Research-Grade Clap Automation (HUD Interface)
============================================================
A futuristic Iron-Man JARVIS-themed GUI backed by the multi-feature
clap detection pipeline defined in jarvis.py.

Usage:
    python jarvis_gui.py
"""

import math
import queue
import threading
import time
import tkinter as tk
import webbrowser
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd

# Import the research-grade detection engine from jarvis.py
from jarvis import (
    ACTIONS,
    ACTION_COOLDOWN,
    CLAP_WINDOW,
    CHUNK_SIZE,
    MIN_CLAP_GAP,
    SAMPLE_RATE,
    ClapCandidate,
    ClapDetector,
    AcousticProfile,
    _rms,
)


# ──────────────────────────────────────────────────────────────
#  Theme Colours
# ──────────────────────────────────────────────────────────────

BG_DARK = "#0a0e17"
BG_PANEL = "#0d1220"
BG_PANEL_BORDER = "#162040"
CYAN = "#00d4ff"
CYAN_DIM = "#004d5c"
CYAN_GLOW = "#00e5ff"
GREEN = "#00ff88"
RED = "#ff3355"
ORANGE = "#ff8c00"
WHITE = "#e0e8f0"
GREY = "#3a4560"
TEXT_DIM = "#5a6a8a"


# ──────────────────────────────────────────────────────────────
#  Data passed from audio thread → GUI
# ──────────────────────────────────────────────────────────────

@dataclass
class AudioEvent:
    kind: str          # "level" | "calibrating" | "calibrated" | "clap" | "pattern" | "action" | "error"
    value: float = 0.0
    text: str = ""
    data: Any = None   # carries AcousticProfile or ClapCandidate


# ──────────────────────────────────────────────────────────────
#  Audio Processing Thread  (uses ClapDetector from jarvis.py)
# ──────────────────────────────────────────────────────────────

class AudioThread(threading.Thread):
    """Background thread: mic capture → ClapDetector → GUI events."""

    def __init__(self, event_queue: queue.Queue, clap_window: float):
        super().__init__(daemon=True)
        self.q = event_queue
        self.clap_window = clap_window
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        detector = ClapDetector(sr=SAMPLE_RATE, chunk=CHUNK_SIZE)

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=CHUNK_SIZE) as stream:

                # ── Calibrate (3-phase) ──
                self.q.put(AudioEvent("calibrating"))

                # Feed level events during calibration for the meter
                original_calibrate = detector.calibrate
                # We need to manually calibrate to send level events
                self._calibrate_with_events(detector, stream)

                profile = detector.profile
                self.q.put(AudioEvent("calibrated", profile.rms_threshold,
                                      "", data=profile))

                # ── Listen ──
                clap_events: list[ClapCandidate] = []
                last_action_time = 0.0

                while not self._stop.is_set():
                    audio, overflowed = stream.read(CHUNK_SIZE)
                    if overflowed:
                        continue

                    # Send audio level for the meter
                    rms = _rms(audio)
                    self.q.put(AudioEvent("level", rms))

                    now = time.time()

                    # Attempt detection
                    candidate = detector.detect(audio)

                    if candidate is not None:
                        # Debounce
                        if clap_events and (now - clap_events[-1].timestamp) < MIN_CLAP_GAP:
                            continue

                        clap_events.append(candidate)
                        self.q.put(AudioEvent("clap", float(len(clap_events)),
                                              "", data=candidate))

                    # Evaluate pattern
                    if clap_events and (now - clap_events[0].timestamp) >= self.clap_window:
                        count = len(clap_events)
                        avg_conf = sum(c.confidence for c in clap_events) / count
                        self.q.put(AudioEvent("pattern", float(count),
                                              f"avg conf {avg_conf:.2f}"))

                        if (now - last_action_time) >= ACTION_COOLDOWN:
                            if count in ACTIONS:
                                label, url = ACTIONS[count]
                                target = url if url else "about:blank"
                                webbrowser.open(target)
                                self.q.put(AudioEvent("action", float(count),
                                                      f"Opened {label}"))
                                last_action_time = time.time()
                        clap_events.clear()

                    # Adaptive threshold
                    detector.adapt()

        except sd.PortAudioError as e:
            self.q.put(AudioEvent("error", text=str(e)))
        except Exception as e:
            self.q.put(AudioEvent("error", text=str(e)))

    def _calibrate_with_events(self, detector: ClapDetector, stream) -> None:
        """Run calibration while feeding level events to the GUI."""
        from jarvis import (
            CALIBRATION_SECONDS, TARGET_PFA,
            extract_features, compute_optimal_spike_factor,
        )

        self.q.put(AudioEvent("level", 0.0))

        rms_v, peak_v, crest_v, zcr_v = [], [], [], []
        sf_v, sc_v, ber_v, kurt_v     = [], [], [], []

        t0 = time.time()
        while time.time() - t0 < CALIBRATION_SECONDS:
            if self._stop.is_set():
                return
            audio, _ = stream.read(CHUNK_SIZE)
            f = extract_features(audio, SAMPLE_RATE)
            rms_v.append(f["rms"]);           peak_v.append(f["peak"])
            crest_v.append(f["crest_factor"]); zcr_v.append(f["zcr"])
            sf_v.append(f["spectral_flatness"])
            sc_v.append(f["spectral_centroid"])
            ber_v.append(f["band_energy_ratio"])
            kurt_v.append(f["kurtosis"])
            # Send level update
            self.q.put(AudioEvent("level", f["rms"]))

        # Populate detector profile
        n = len(rms_v)
        p = detector.profile
        p.sample_count       = n
        p.ambient_rms_mean   = max(float(np.mean(rms_v)),  1e-9)
        p.ambient_rms_std    = max(float(np.std(rms_v, ddof=1 if n > 1 else 0)), 1e-10)
        p.ambient_rms_median = float(np.median(rms_v))
        p.ambient_rms_p999   = float(np.percentile(rms_v, 99.9))
        p.ambient_peak_mean  = float(np.mean(peak_v))
        p.ambient_crest_mean = float(np.mean(crest_v))
        p.ambient_crest_std  = max(float(np.std(crest_v, ddof=1 if n > 1 else 0)), 0.01)
        p.ambient_zcr_mean   = float(np.mean(zcr_v))
        p.ambient_sf_mean    = float(np.mean(sf_v))
        p.ambient_sc_mean    = float(np.mean(sc_v))
        p.ambient_kurtosis   = float(np.mean(kurt_v))

        factor, f_np, f_emp, f_pct = compute_optimal_spike_factor(
            rms_v, peak_v, crest_v, kurt_v, target_pfa=TARGET_PFA,
        )
        p.spike_factor              = factor
        p.factor_neyman_pearson     = f_np
        p.factor_empirical_snr      = f_emp
        p.factor_percentile         = f_pct
        p.rms_threshold             = p.ambient_rms_mean * factor
        p.min_crest_factor          = max(p.ambient_crest_mean + 2 * p.ambient_crest_std, 3.0)
        p.min_spectral_flat         = 0.12

        detector._quiet_rms.extend(rms_v)
        detector._last_adapt = time.time()
        detector.calibrated = True


# ──────────────────────────────────────────────────────────────
#  GUI Application
# ──────────────────────────────────────────────────────────────

class JarvisApp:
    WIDTH = 960
    HEIGHT = 680

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("J.A.R.V.I.S — Research-Grade Clap Automation")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)
        self.root.geometry(f"{self.WIDTH}x{self.HEIGHT}")

        # ── State ──
        self.running = False
        self.audio_thread: AudioThread | None = None
        self.event_queue: queue.Queue[AudioEvent] = queue.Queue()
        self.audio_level = 0.0
        self.clap_count = 0
        self.arc_angle = 0.0
        self.status_text = "OFFLINE"
        self.status_colour = GREY
        self.threshold = 0.0
        self.pulse_phase = 0.0
        self.confidence = 0.0  # latest clap confidence

        self._build_ui()
        self._animate()

    # ──────────────────────────────────────────
    #  UI Construction
    # ──────────────────────────────────────────

    def _build_ui(self):
        self.canvas = tk.Canvas(self.root, width=self.WIDTH, height=self.HEIGHT,
                                bg=BG_DARK, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # ── Left Panel: Action Mapping ──
        self._draw_panel(20, 20, 240, 230, "ACTION MAP")

        actions_text = [
            ("① Single Clap", "→  ChatGPT"),
            ("② Double Clap", "→  YouTube"),
            ("③ Triple Clap", "→  Homepage"),
        ]
        for i, (label, target) in enumerate(actions_text):
            y = 75 + i * 52
            self.canvas.create_text(35, y, text=label, fill=CYAN, anchor="w",
                                    font=("Consolas", 11, "bold"))
            self.canvas.create_text(35, y + 20, text=target, fill=TEXT_DIM, anchor="w",
                                    font=("Consolas", 10))

        # ── Left Panel: Diagnostics ──
        self._draw_panel(20, 268, 240, 210, "DIAGNOSTICS")

        dy = 318
        self.diag_labels = {}
        diag_items = [
            ("spike",     "Spike Factor : --"),
            ("threshold", "Threshold    : --"),
            ("crest",     "Min Crest    : --"),
            ("f_np",      "F. Neyman-P  : --"),
            ("f_emp",     "F. Empirical : --"),
            ("f_pct",     "F. Percentile: --"),
            ("ambient",   "Ambient RMS  : --"),
            ("chunks",    "Chunks       : --"),
        ]
        for key, text in diag_items:
            self.diag_labels[key] = self.canvas.create_text(
                35, dy, text=text, fill=TEXT_DIM, anchor="w", font=("Consolas", 8))
            dy += 22

        # ── Centre: HUD Ring Area ──
        self.hud_cx = 490
        self.hud_cy = 220
        self.hud_r = 115

        # ── Centre: Clap Counter ──
        self.clap_label = self.canvas.create_text(
            self.hud_cx, self.hud_cy - 15, text="0", fill=CYAN,
            font=("Consolas", 60, "bold"))
        self.clap_sublabel = self.canvas.create_text(
            self.hud_cx, self.hud_cy + 35, text="CLAPS", fill=TEXT_DIM,
            font=("Consolas", 12))

        # ── Centre: Confidence Display ──
        self.conf_label = self.canvas.create_text(
            self.hud_cx, self.hud_cy + 58, text="", fill=CYAN_DIM,
            font=("Consolas", 10))

        # ── Centre: Status Badge ──
        self.status_badge = self.canvas.create_text(
            self.hud_cx, self.hud_cy + 160, text="OFFLINE",
            fill=GREY, font=("Consolas", 14, "bold"))

        # ── Centre: Audio Level Meter ──
        meter_x = self.hud_cx - 130
        meter_y = self.hud_cy + 120
        meter_w = 260
        meter_h = 10
        self.canvas.create_rectangle(meter_x, meter_y, meter_x + meter_w, meter_y + meter_h,
                                     fill=BG_PANEL, outline=GREY, width=1)
        self.level_bar = self.canvas.create_rectangle(
            meter_x + 1, meter_y + 1, meter_x + 2, meter_y + meter_h - 1,
            fill=CYAN, outline="")
        self.meter_x, self.meter_w = meter_x, meter_w
        self.meter_y, self.meter_h = meter_y, meter_h

        self.canvas.create_text(meter_x, meter_y - 12, text="AUDIO LEVEL",
                                fill=TEXT_DIM, anchor="w", font=("Consolas", 8))

        # ── Right Panel: Activity Log ──
        self._draw_panel(700, 20, 240, 458, "ACTIVITY LOG")

        self.log_frame = tk.Frame(self.root, bg=BG_PANEL, highlightthickness=0)
        self.log_text = tk.Text(self.log_frame, bg=BG_PANEL, fg=TEXT_DIM,
                                font=("Consolas", 8), wrap="word",
                                borderwidth=0, highlightthickness=0,
                                insertbackground=BG_PANEL, state="disabled",
                                padx=6, pady=4)
        self.log_text.tag_configure("cyan", foreground=CYAN)
        self.log_text.tag_configure("green", foreground=GREEN)
        self.log_text.tag_configure("red", foreground=RED)
        self.log_text.tag_configure("orange", foreground=ORANGE)
        self.log_text.tag_configure("dim", foreground=TEXT_DIM)
        self.log_text.tag_configure("white", foreground=WHITE)

        self.log_text.pack(fill="both", expand=True)
        self.canvas.create_window(715, 58, window=self.log_frame,
                                  width=210, height=405, anchor="nw")

        # ── Bottom Bar: Controls ──
        self._draw_panel(20, 498, 920, 160, "CONTROLS")

        # Start / Stop button
        self.btn_start = tk.Button(
            self.root, text="▶  START LISTENING", font=("Consolas", 13, "bold"),
            bg=BG_PANEL, fg=CYAN, activebackground="#102040", activeforeground=CYAN_GLOW,
            borderwidth=0, padx=24, pady=10, cursor="hand2",
            command=self._toggle_listening)
        self.canvas.create_window(200, 575, window=self.btn_start)

        # Clap Window slider
        self.canvas.create_text(480, 540, text="CLAP WINDOW (s)", fill=TEXT_DIM,
                                font=("Consolas", 9))
        self.window_var = tk.DoubleVar(value=CLAP_WINDOW)
        self.window_slider = tk.Scale(
            self.root, from_=0.3, to=3.0, resolution=0.05, orient="horizontal",
            variable=self.window_var, length=200, bg=BG_PANEL, fg=CYAN,
            troughcolor=BG_DARK, highlightthickness=0, borderwidth=0,
            font=("Consolas", 9), activebackground=CYAN_DIM)
        self.canvas.create_window(580, 578, window=self.window_slider)

        # Info label
        self.canvas.create_text(480, 620, text="Spike factor is auto-computed by fusing",
                                fill=TEXT_DIM, font=("Consolas", 8), anchor="w")
        self.canvas.create_text(480, 635, text="Neyman–Pearson / Empirical SNR / Percentile",
                                fill=CYAN_DIM, font=("Consolas", 8), anchor="w")

        # ── Title ──
        self.canvas.create_text(self.hud_cx, 14, text="J . A . R . V . I . S",
                                fill=CYAN, font=("Consolas", 11, "bold"), anchor="n")
        self.canvas.create_text(self.hud_cx, 32,
                                text="RESEARCH-GRADE  DETECTION",
                                fill=GREY, font=("Consolas", 8), anchor="n")

        # ── Watermark ──
        self.canvas.create_text(self.WIDTH - 10, self.HEIGHT - 8,
                                text="v2.0  •  Multi-Feature Pipeline", fill="#1a2440",
                                font=("Consolas", 8), anchor="se")

    def _draw_panel(self, x, y, w, h, title=""):
        self.canvas.create_rectangle(x, y, x + w, y + h,
                                     fill=BG_PANEL, outline=BG_PANEL_BORDER, width=1)
        if title:
            self.canvas.create_text(x + 12, y + 14, text=title, fill=GREY,
                                    anchor="w", font=("Consolas", 9, "bold"))
            self.canvas.create_line(x + 8, y + 30, x + w - 8, y + 30,
                                    fill=BG_PANEL_BORDER)

    # ──────────────────────────────────────────
    #  HUD Arc Drawing
    # ──────────────────────────────────────────

    def _draw_hud_ring(self):
        self.canvas.delete("hud_arc")
        cx, cy, r = self.hud_cx, self.hud_cy, self.hud_r

        glow = 0.6 + 0.4 * math.sin(self.pulse_phase)

        ring_specs = [
            (r + 25, 3, 0, 290, CYAN_DIM),
            (r + 25, 3, 310, 40, CYAN_DIM),
            (r + 15, 2, 30, 120, CYAN),
            (r + 15, 2, 180, 90, CYAN),
            (r + 5, 1.5, 0, 360, CYAN_DIM),
            (r - 5, 1, 90, 180, GREY),
        ]

        for radius, width, start_offset, extent, colour in ring_specs:
            start = start_offset + self.arc_angle
            bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
            self.canvas.create_arc(*bbox, start=start, extent=extent,
                                   style="arc", outline=colour,
                                   width=width, tags="hud_arc")

        # Rotating accent arcs
        for i in range(3):
            angle = self.arc_angle * 1.5 + i * 120
            accent_r = r + 20
            bbox = (cx - accent_r, cy - accent_r, cx + accent_r, cy + accent_r)
            c = self._lerp_colour(CYAN_DIM, CYAN_GLOW, glow)
            self.canvas.create_arc(*bbox, start=angle, extent=25,
                                   style="arc", outline=c,
                                   width=2.5, tags="hud_arc")

        # Tick marks
        tick_r = r + 32
        for i in range(36):
            a = math.radians(i * 10 + self.arc_angle * 0.3)
            x1 = cx + tick_r * math.cos(a)
            y1 = cy - tick_r * math.sin(a)
            x2 = cx + (tick_r + 4) * math.cos(a)
            y2 = cy - (tick_r + 4) * math.sin(a)
            self.canvas.create_line(x1, y1, x2, y2, fill=CYAN_DIM,
                                    width=1, tags="hud_arc")

        # Confidence arc (inner ring that fills based on latest confidence)
        if self.confidence > 0:
            conf_r = r - 15
            conf_extent = self.confidence * 360
            bbox = (cx - conf_r, cy - conf_r, cx + conf_r, cy + conf_r)
            conf_colour = self._lerp_colour(ORANGE, GREEN, self.confidence)
            self.canvas.create_arc(*bbox, start=90, extent=conf_extent,
                                   style="arc", outline=conf_colour,
                                   width=3, tags="hud_arc")

    @staticmethod
    def _lerp_colour(c1: str, c2: str, t: float) -> str:
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    # ──────────────────────────────────────────
    #  Animation / Event Loop
    # ──────────────────────────────────────────

    def _animate(self):
        self.arc_angle = (self.arc_angle + 0.6) % 360
        self.pulse_phase += 0.08

        self._draw_hud_ring()
        self._update_level_bar()
        self._poll_events()

        self.root.after(33, self._animate)

    def _update_level_bar(self):
        if self.threshold > 0:
            norm = min(self.audio_level / (self.threshold * 2), 1.0)
        else:
            norm = min(self.audio_level * 5000, 1.0)

        bar_w = max(2, int(norm * (self.meter_w - 2)))
        x1 = self.meter_x + 1
        y1 = self.meter_y + 1
        x2 = self.meter_x + 1 + bar_w
        y2 = self.meter_y + self.meter_h - 1

        if norm < 0.4:
            colour = CYAN
        elif norm < 0.7:
            colour = GREEN
        elif norm < 0.9:
            colour = ORANGE
        else:
            colour = RED

        self.canvas.coords(self.level_bar, x1, y1, x2, y2)
        self.canvas.itemconfig(self.level_bar, fill=colour)

        self.audio_level *= 0.85
        self.confidence *= 0.97  # slow decay for confidence arc

    def _poll_events(self):
        try:
            while True:
                event = self.event_queue.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass

    def _handle_event(self, event: AudioEvent):
        if event.kind == "level":
            self.audio_level = max(self.audio_level, event.value)

        elif event.kind == "calibrating":
            self._set_status("CALIBRATING…", ORANGE)
            self._log("🎚️  Calibrating — stay quiet…", "orange")
            self._log("   3-phase: sample / compute / verify", "dim")

        elif event.kind == "calibrated":
            profile: AcousticProfile = event.data
            self.threshold = profile.rms_threshold
            self._set_status("LISTENING", GREEN)

            # Update diagnostics panel
            self.canvas.itemconfig(self.diag_labels["spike"],
                                   text=f"Spike Factor : {profile.spike_factor:.4f}")
            self.canvas.itemconfig(self.diag_labels["threshold"],
                                   text=f"Threshold    : {profile.rms_threshold:.8f}")
            self.canvas.itemconfig(self.diag_labels["crest"],
                                   text=f"Min Crest    : {profile.min_crest_factor:.2f}")
            self.canvas.itemconfig(self.diag_labels["f_np"],
                                   text=f"F. Neyman-P  : {profile.factor_neyman_pearson:.4f}")
            self.canvas.itemconfig(self.diag_labels["f_emp"],
                                   text=f"F. Empirical : {profile.factor_empirical_snr:.4f}")
            self.canvas.itemconfig(self.diag_labels["f_pct"],
                                   text=f"F. Percentile: {profile.factor_percentile:.4f}")
            self.canvas.itemconfig(self.diag_labels["ambient"],
                                   text=f"Ambient RMS  : {profile.ambient_rms_mean:.8f}")
            self.canvas.itemconfig(self.diag_labels["chunks"],
                                   text=f"Chunks       : {profile.sample_count}")

            self._log(f"✅ Spike={profile.spike_factor:.2f}  "
                      f"Thr={profile.rms_threshold:.6f}", "green")
            self._log(f"   NP={profile.factor_neyman_pearson:.2f}  "
                      f"Emp={profile.factor_empirical_snr:.2f}  "
                      f"Pct={profile.factor_percentile:.2f}", "dim")
            self._log("🔇 Listening for claps…", "cyan")

        elif event.kind == "clap":
            candidate: ClapCandidate = event.data
            self.clap_count = int(event.value)
            self.confidence = candidate.confidence
            self.canvas.itemconfig(self.clap_label, text=str(self.clap_count))
            self.canvas.itemconfig(self.conf_label,
                                   text=f"CONF {candidate.confidence:.0%}")
            self._set_status("CLAP DETECTED!", CYAN_GLOW)
            self._log(
                f"🔊 Clap #{self.clap_count}  "
                f"C={candidate.confidence:.2f}  "
                f"CF={candidate.crest_factor:.1f}  "
                f"SF={candidate.spectral_flatness:.2f}",
                "cyan"
            )
            self.root.after(600, lambda: self._set_status("LISTENING", GREEN)
                            if self.running else None)

        elif event.kind == "pattern":
            count = int(event.value)
            self._log(f"🎯 Pattern: {count} clap(s) ({event.text})", "white")

        elif event.kind == "action":
            self._log(f"✨ {event.text}", "green")
            self.root.after(1500, self._reset_counter)

        elif event.kind == "error":
            self._set_status("ERROR", RED)
            self._log(f"❌ {event.text}", "red")
            self.running = False
            self.btn_start.configure(text="▶  START LISTENING", fg=CYAN)

    def _set_status(self, text: str, colour: str):
        self.status_text = text
        self.status_colour = colour
        self.canvas.itemconfig(self.status_badge, text=text, fill=colour)

    def _reset_counter(self):
        self.clap_count = 0
        self.confidence = 0.0
        self.canvas.itemconfig(self.clap_label, text="0")
        self.canvas.itemconfig(self.conf_label, text="")

    def _log(self, message: str, tag: str = "dim"):
        ts = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts}] ", "dim")
        self.log_text.insert("end", f"{message}\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ──────────────────────────────────────────
    #  Controls
    # ──────────────────────────────────────────

    def _toggle_listening(self):
        if self.running:
            self._stop_listening()
        else:
            self._start_listening()

    def _start_listening(self):
        self.running = True
        clap_window = self.window_var.get()

        self.btn_start.configure(text="■  STOP", fg=RED)
        self._set_status("STARTING…", ORANGE)
        self._log(f"▶ Starting (window={clap_window}s)…", "cyan")

        self.event_queue = queue.Queue()
        self.audio_thread = AudioThread(self.event_queue, clap_window)
        self.audio_thread.start()

    def _stop_listening(self):
        self.running = False
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread = None
        self.btn_start.configure(text="▶  START LISTENING", fg=CYAN)
        self._set_status("OFFLINE", GREY)
        self._log("⏹ Stopped.", "dim")
        self._reset_counter()

    def on_close(self):
        if self.audio_thread:
            self.audio_thread.stop()
        self.root.destroy()


# ──────────────────────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()

    # Dark title-bar on Windows 10/11
    try:
        from ctypes import windll, byref, c_int, sizeof
        hwnd = windll.user32.GetParent(root.winfo_id())
        windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, byref(c_int(1)), sizeof(c_int))
    except Exception:
        pass

    app = JarvisApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
