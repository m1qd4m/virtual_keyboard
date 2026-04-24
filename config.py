"""config.py — Tuned for maximum accuracy."""


class Config:
    def __init__(self, args=None):
        self.cam_index       = getattr(args, "cam",      0)
        self.frame_width     = getattr(args, "width",    1280)
        self.frame_height    = getattr(args, "height",   720)
        self.target_fps      = getattr(args, "fps",      30)

        # MediaPipe confidence
        self.detection_conf  = 0.80
        self.tracking_conf   = 0.70

        # Smoothing — very low alpha = very smooth/stable cursor
        self.ema_alpha       = 0.12

        # Click — pinch distance in pixels
        # Lower = harder to click (more intentional)
        # Higher = easier but accidental
        self.pinch_threshold = 65.0

        # How many frames pinch must be HELD before click fires
        # Higher = more deliberate, fewer accidents
        self.pinch_hold_frames = 3

        # Cooldown between clicks in ms
        self.debounce_ms     = 600

        # Key snap radius — 0 to disable snapping entirely
        self.snap_radius     = 0      # OFF — exact hit only for accuracy

        # Swipe disabled — removed for accuracy focus
        self.swipe_min_keys  = 999

        # UI
        self.show_landmarks  = True
        self.theme           = "dark"
        self.sound_enabled   = not getattr(args, "no_sound", False)

        # Keyboard position (0=top, 1=bottom)
        self.keyboard_y_frac = 0.38