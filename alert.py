"""
alert.py
Handles audio and visual alerts when drowsiness is detected.
Uses pygame for sound (falls back to beep if unavailable).
"""

import time
import threading

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False


class AlertSystem:
    """
    Manages drowsiness alerts with cooldown to avoid alarm fatigue.

    Parameters
    ----------
    sound_enabled : bool   — Play audio alert (default True)
    cooldown_sec  : float  — Seconds between repeated alerts (default 2.0)
    """

    ALERT_FILE   = "sounds/alert.wav"
    WARNING_FILE = "sounds/warning.wav"

    def __init__(self, sound_enabled=True, cooldown_sec=2.0):
        self.sound_enabled = sound_enabled and PYGAME_AVAILABLE
        self.cooldown      = cooldown_sec
        self._last_alert   = 0
        self._active       = False

        if sound_enabled and not PYGAME_AVAILABLE:
            print("[WARNING] pygame not installed. Sound alerts disabled.")
            print("         Install with: pip install pygame")

        self._load_sounds()

    def _load_sounds(self):
        self._alert_sound   = None
        self._warning_sound = None

        if not self.sound_enabled:
            return

        try:
            self._alert_sound = pygame.mixer.Sound(self.ALERT_FILE)
            self._alert_sound.set_volume(1.0)
        except Exception:
            print(f"[WARNING] Alert sound not found: {self.ALERT_FILE}")
            print("          Generating beep tone instead...")
            self._alert_sound = self._generate_beep(880, 0.5)

        try:
            self._warning_sound = pygame.mixer.Sound(self.WARNING_FILE)
            self._warning_sound.set_volume(0.7)
        except Exception:
            self._warning_sound = self._generate_beep(440, 0.3)

    def _generate_beep(self, frequency=880, duration=0.5):
        """Generate a simple sine-wave beep using pygame."""
        if not PYGAME_AVAILABLE:
            return None
        try:
            import numpy as np
            sample_rate = 44100
            samples     = int(sample_rate * duration)
            t           = np.linspace(0, duration, samples, False)
            wave        = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
            stereo      = np.column_stack([wave, wave])
            sound       = pygame.sndarray.make_sound(stereo)
            return sound
        except Exception:
            return None

    def trigger(self, level="danger"):
        """
        Fire an alert.

        Parameters
        ----------
        level : "danger"  — Eyes closed / drowsy (loud alert)
                "warning" — Yawning detected     (soft warning)
        """
        now = time.time()
        if now - self._last_alert < self.cooldown:
            return                             # Still in cooldown

        self._active    = True
        self._last_alert = now

        if self.sound_enabled:
            sound = self._alert_sound if level == "danger" else self._warning_sound
            if sound:
                threading.Thread(target=sound.play, daemon=True).start()
        else:
            # Terminal bell fallback
            print("\a", end="", flush=True)

        level_label = "🚨 DROWSY ALERT" if level == "danger" else "⚠️  YAWN DETECTED"
        print(f"[ALERT] {level_label} at {time.strftime('%H:%M:%S')}")

    def reset(self):
        """Call this when the driver is awake again."""
        self._active = False

    @property
    def is_active(self):
        return self._active
