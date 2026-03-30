"""Diagnostic: shows what each detection gate does with your audio."""
import time
import numpy as np
import sounddevice as sd
from jarvis import (
    SAMPLE_RATE, CHUNK_SIZE, CALIBRATION_SECONDS,
    ClapDetector, extract_features, _rms
)

detector = ClapDetector(sr=SAMPLE_RATE, chunk=CHUNK_SIZE)

print("=== GATE DIAGNOSTIC ===")
print("Calibrating (stay quiet)...")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                    dtype="float32", blocksize=CHUNK_SIZE) as stream:
    detector.calibrate(stream)
    p = detector.profile

    print(f"\nThreshold: {p.rms_threshold:.8f}")
    print(f"Min Crest: {p.min_crest_factor:.2f}")
    print(f"Min Spec Flat: {p.min_spectral_flat}")
    print(f"Spike Factor: {p.spike_factor:.4f}")
    print(f"\nNow CLAP! Monitoring for 15s — showing ALL chunks above ambient...\n")

    start = time.time()
    while time.time() - start < 15:
        audio, _ = stream.read(CHUNK_SIZE)
        feat = extract_features(audio, SAMPLE_RATE)
        rms = feat["rms"]

        # Show anything above 50% of threshold (near-misses too)
        if rms > p.rms_threshold * 0.3:
            gate1 = "PASS" if rms > p.rms_threshold else "FAIL"
            gate2 = "PASS" if feat["crest_factor"] > p.min_crest_factor else "FAIL"
            gate3 = "PASS" if feat["spectral_flatness"] > p.min_spectral_flat else "FAIL"

            result = detector.detect(audio)
            final = f"CONF={result.confidence:.2f}" if result else "REJECTED"

            elapsed = time.time() - start
            print(
                f"[{elapsed:5.1f}s] "
                f"RMS={rms:.6f} ({gate1})  "
                f"CF={feat['crest_factor']:.1f} ({gate2})  "
                f"SF={feat['spectral_flatness']:.3f} ({gate3})  "
                f"SC={feat['spectral_centroid']:.0f}Hz  "
                f"Kurt={feat['kurtosis']:.1f}  "
                f"→ {final}"
            )

print("\nDone.")
