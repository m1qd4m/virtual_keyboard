"""gesture_logic.py — Click detection only. Swipe removed.

Click algorithm
---------------
1. Index tip must be INSIDE a key bounding box (no snap).
2. Pinch distance must be below threshold for N consecutive frames.
3. Debounce prevents re-firing until cooldown expires.
4. On release (pinch opens), system resets — ready for next key.
"""

import time
from typing import List, Dict, Optional


class GestureEngine:

    def __init__(self, cfg):
        self.cfg             = cfg
        self._last_click     = 0.0
        self._pinch_frames   = 0      # consecutive frames pinch is active
        self._click_fired    = False  # True after click fired, reset when pinch releases
        self._hovered_idx    : Optional[int] = None
        self._swipe_path     : List = []   # kept for trail drawing only

    def update(self, hand_data, key_regions) -> List[Dict]:
        events = []

        if not hand_data.detected:
            self._pinch_frames = 0
            self._click_fired  = False
            self._hovered_idx  = None
            self._swipe_path   = []
            return events

        ix, iy       = hand_data.index_tip
        pinch_active = hand_data.pinch_dist < self.cfg.pinch_threshold

        # ── Find hovered key — exact hit ONLY ────────────────────────────────
        hovered_key       = None
        self._hovered_idx = None
        for k in key_regions:
            if k.contains(ix, iy):
                hovered_key       = k
                self._hovered_idx = k.index
                break

        now = time.time()

        # ── Pinch state machine ───────────────────────────────────────────────
        if pinch_active:
            self._pinch_frames += 1
        else:
            # Pinch released — reset so next pinch can fire
            self._pinch_frames = 0
            self._click_fired  = False

        debounce_ok = (now - self._last_click) > (self.cfg.debounce_ms / 1000.0)

        # Fire when:
        #   - pinch held for required frames
        #   - click not already fired this pinch
        #   - debounce passed
        #   - finger is on a key
        if (self._pinch_frames >= self.cfg.pinch_hold_frames
                and not self._click_fired
                and debounce_ok
                and hovered_key is not None):
            events.append({
                "type":      "key_press",
                "key":       hovered_key.label,
                "key_index": hovered_key.index,
            })
            self._last_click   = now
            self._click_fired  = True   # block until pinch releases

        # Trail for cursor rendering
        self._swipe_path.append((ix, iy))
        if len(self._swipe_path) > 30:
            self._swipe_path.pop(0)

        return events

    def get_swipe_path(self) -> List:
        return self._swipe_path

    def get_hovered_idx(self) -> Optional[int]:
        return self._hovered_idx