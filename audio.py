"""audio.py — Cross-platform click sound feedback."""

import sys
import threading


class AudioFeedback:
    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._backend = "none"

        if sys.platform == "win32":
            self._backend = "win"
        else:
            try:
                import numpy as np
                import simpleaudio as sa
                self._backend = "sa"
                self._sa = sa
                self._click_wave = self._make_wave()
            except ImportError:
                self._backend = "none"

    def play_click(self):
        if not self._enabled:
            return
        threading.Thread(target=self._play, daemon=True).start()

    def toggle(self):
        self._enabled = not self._enabled

    def _play(self):
        try:
            if self._backend == "win":
                import winsound
                winsound.Beep(880, 40)
            elif self._backend == "sa":
                self._sa.play_buffer(self._click_wave, 1, 2, 44100)
        except Exception:
            pass

    def _make_wave(self):
        import numpy as np
        t = np.linspace(0, 0.045, int(44100 * 0.045), endpoint=False)
        w = (np.sin(2 * np.pi * 880 * t) * 0.25 * 32767).astype(np.int16)
        fade = np.linspace(1, 0, len(w))
        return (w * fade).astype(np.int16)
