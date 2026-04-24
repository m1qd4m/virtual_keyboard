"""hand_tracking.py — Tracks ONLY index tip + thumb tip.
All other fingers are intentionally ignored.
Compatible with mediapipe 0.10.13+
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Hardcoded — never need mp.solutions
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


@dataclass
class HandData:
    detected     : bool  = False
    landmarks_px : List  = field(default_factory=list)
    index_tip    : Optional[Tuple[int,int]] = None   # landmark 8
    thumb_tip    : Optional[Tuple[int,int]] = None   # landmark 4
    pinch_dist   : float = 0.0
    confidence   : float = 0.0


class EMAFilter:
    """Separate EMA smoother for index tip and thumb tip only."""
    def __init__(self, alpha: float = 0.12):
        self.alpha = alpha
        self._ix = self._iy = None
        self._tx = self._ty = None

    def update_index(self, x, y):
        if self._ix is None:
            self._ix, self._iy = float(x), float(y)
        else:
            self._ix = self.alpha * x + (1 - self.alpha) * self._ix
            self._iy = self.alpha * y + (1 - self.alpha) * self._iy
        return int(self._ix), int(self._iy)

    def update_thumb(self, x, y):
        if self._tx is None:
            self._tx, self._ty = float(x), float(y)
        else:
            self._tx = self.alpha * x + (1 - self.alpha) * self._tx
            self._ty = self.alpha * y + (1 - self.alpha) * self._ty
        return int(self._tx), int(self._ty)

    def reset(self):
        self._ix = self._iy = None
        self._tx = self._ty = None


class HandTracker:
    THUMB_TIP  = 4
    INDEX_TIP  = 8

    def __init__(self, cfg):
        self.cfg         = cfg
        self.smoother    = EMAFilter(alpha=cfg.ema_alpha)
        self._hands      = None
        self._use_legacy = False
        self._all_lm_pts = []   # all 21 pts for drawing only
        self._setup(cfg)

    def _setup(self, cfg):
        # Try legacy API
        try:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode        = False,
                max_num_hands            = 1,
                min_detection_confidence = cfg.detection_conf,
                min_tracking_confidence  = cfg.tracking_conf,
                model_complexity         = 0,
            )
            self._use_legacy = True
            print("[INFO] MediaPipe: legacy solutions.hands")
            return
        except AttributeError:
            pass

        # Tasks API fallback
        print("[INFO] Using MediaPipe Tasks API...")
        import mediapipe as mp, urllib.request, os, tempfile
        model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
        if not os.path.exists(model_path):
            print("[INFO] Downloading hand_landmarker.task (~5MB)...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                model_path)
            print("[INFO] Downloaded.")
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode = mp.tasks.vision.RunningMode.IMAGE,
            num_hands    = 1,
            min_hand_detection_confidence = cfg.detection_conf,
            min_hand_presence_confidence  = cfg.tracking_conf,
            min_tracking_confidence       = cfg.tracking_conf,
        )
        self._hands      = mp.tasks.vision.HandLandmarker.create_from_options(opts)
        self._use_legacy = False
        print("[INFO] MediaPipe Tasks ready.")

    def process(self, frame: np.ndarray) -> HandData:
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._use_legacy:
            rgb.flags.writeable = False
            res = self._hands.process(rgb)
            rgb.flags.writeable = True
            if not res.multi_hand_landmarks:
                self.smoother.reset()
                self._all_lm_pts = []
                return HandData()
            lms  = res.multi_hand_landmarks[0].landmark
            conf = res.multi_handedness[0].classification[0].score
        else:
            import mediapipe as mp
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = self._hands.detect(mp_img)
            if not res.hand_landmarks:
                self.smoother.reset()
                self._all_lm_pts = []
                return HandData()
            lms  = res.hand_landmarks[0]
            conf = res.handedness[0][0].score if res.handedness else 1.0

        # All 21 landmarks for skeleton drawing
        self._all_lm_pts = [(int(lm.x*w), int(lm.y*h)) for lm in lms]

        # Smooth ONLY index tip and thumb tip
        raw_ix = lms[self.INDEX_TIP].x * w
        raw_iy = lms[self.INDEX_TIP].y * h
        raw_tx = lms[self.THUMB_TIP].x * w
        raw_ty = lms[self.THUMB_TIP].y * h

        idx = self.smoother.update_index(raw_ix, raw_iy)
        thm = self.smoother.update_thumb(raw_tx, raw_ty)

        return HandData(
            detected     = True,
            landmarks_px = self._all_lm_pts,
            index_tip    = idx,
            thumb_tip    = thm,
            pinch_dist   = float(np.hypot(idx[0]-thm[0], idx[1]-thm[1])),
            confidence   = conf,
        )

    def close(self):
        if self._hands:
            try: self._hands.close()
            except: pass