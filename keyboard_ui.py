"""keyboard_ui.py — Clean keyboard UI focused on accuracy.

Key design decisions
--------------------
- Keys are larger, with generous padding between them
- Corner keys (Q, P, Z, M, CAPS) are same size as others — no shrinking
- Keyboard is centered with equal margins on all sides
- No word suggestions bar (removed as requested)
- Cursor shows ONLY index tip and thumb tip — large and obvious
- Pinch meter always visible so user knows state at a glance
- Press flash is very obvious (bright + long duration)
- Hover state is clearly distinct from normal state
"""

import cv2
import numpy as np
import time
from typing import List, Optional

THEMES = {
    "dark": {
        "bg_overlay":   (10,  12,  22,  180),
        "key_bg":       (28,  32,  48,  160),
        "key_border":   (80,  110, 200, 160),
        "key_text":     (230, 238, 255),
        "hover_bg":     (30,  130, 255, 220),
        "hover_border": (0,   200, 255, 255),
        "press_bg":     (0,   230, 180, 245),
        "press_border": (0,   255, 200, 255),
        "text_bar_bg":  (14,  16,  28,  220),
        "text_color":   (215, 232, 255),
        "fps_color":    (70,  245, 130),
        "special_bg":   (55,  18,  95,  185),
        "back_bg":      (130, 18,  55,  190),
        "clear_bg":     (18,  70,  155, 190),
        "meter_empty":  (35,  38,  58),
        "meter_fill":   (0,   210, 160),
        "meter_full":   (0,   255, 200),
        "meter_border": (70,  80,  130),
    },
    "light": {
        "bg_overlay":   (200, 210, 230, 160),
        "key_bg":       (218, 228, 248, 175),
        "key_border":   (85,  105, 195, 180),
        "key_text":     (18,  22,  52),
        "hover_bg":     (80,  140, 255, 210),
        "hover_border": (0,   160, 220, 255),
        "press_bg":     (50,  195, 150, 235),
        "press_border": (0,   210, 160, 255),
        "text_bar_bg":  (225, 232, 250, 215),
        "text_color":   (14,  18,  48),
        "fps_color":    (25,  105, 25),
        "special_bg":   (182, 192, 228, 175),
        "back_bg":      (210, 125, 148, 190),
        "clear_bg":     (135, 175, 232, 190),
        "meter_empty":  (195, 205, 225),
        "meter_fill":   (30,  160, 120),
        "meter_full":   (0,   190, 140),
        "meter_border": (140, 155, 200),
    },
}

ROWS = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["CAPS","Z","X","C","V","B","N","M"],
    ["BACK","SPACE","CLEAR","ENTER"],
]
SPECIAL_KEYS    = {"CAPS","BACK","SPACE","ENTER","CLEAR"}
BOTTOM_WIDTHS   = {"BACK": 1.5, "SPACE": 3.0, "CLEAR": 1.8, "ENTER": 1.8}
KEY_LABELS      = {"BACK":"DEL", "CLEAR":"CLEAR ALL", "ENTER":"ENTER", "CAPS":"CAPS",
                   "SPACE":"SPACE"}


class Key:
    __slots__ = ("label","x","y","w","h","index","press_t","is_special")

    def __init__(self, label, x, y, w, h, index):
        self.label      = label
        self.x          = x
        self.y          = y
        self.w          = w
        self.h          = h
        self.index      = index
        self.press_t    = 0.0
        self.is_special = label in SPECIAL_KEYS

    def contains(self, px, py) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    @property
    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)


class KeyboardUI:
    ANIM_PRESS_DUR  = 0.35    # seconds press flash stays visible
    BLUR_KSIZE      = 17
    TEXT_BAR_H      = 72
    KEY_RADIUS      = 10
    CURSOR_BLINK    = 0.6
    KEY_PAD_H       = 8       # horizontal padding between keys
    KEY_PAD_V       = 7       # vertical padding between rows

    def __init__(self, frame_w: int, frame_h: int, cfg):
        self.fw         = frame_w
        self.fh         = frame_h
        self.cfg        = cfg
        self.theme_name = "dark"
        self.theme      = THEMES["dark"]
        self.caps_lock  = False
        self._keys      : List[Key] = []
        self._build_layout()
        self._cursor_t  = time.time()

    # ── Layout ─────────────────────────────────────────────────────────────────
    def _build_layout(self):
        margin   = int(self.fw * 0.025)
        kb_w     = self.fw - 2 * margin
        n_rows   = len(ROWS)
        key_h    = 58    # fixed key height in pixels
        kb_h     = n_rows * key_h + (n_rows + 1) * self.KEY_PAD_V
        kb_x     = margin
        # Position: use config fraction, clamp so it fits
        kb_y     = int(self.fh * self.cfg.keyboard_y_frac)
        kb_y     = min(kb_y, self.fh - kb_h - 4)
        kb_y     = max(self.TEXT_BAR_H + 10, kb_y)

        self.kb_rect = (kb_x, kb_y, kb_w, kb_h)
        idx          = 0
        self._keys.clear()

        for r, row in enumerate(ROWS):
            y = kb_y + self.KEY_PAD_V + r * (key_h + self.KEY_PAD_V)

            if r < 3:
                # Equal-width keys
                n     = len(row)
                avail = kb_w - self.KEY_PAD_H * (n + 1)
                kw    = avail // n
                x     = kb_x + self.KEY_PAD_H
                for label in row:
                    self._keys.append(Key(label, x, y, kw, key_h, idx))
                    x   += kw + self.KEY_PAD_H
                    idx += 1
            else:
                # Bottom row — proportional widths
                total  = sum(BOTTOM_WIDTHS.get(k, 1.0) for k in row)
                avail  = kb_w - self.KEY_PAD_H * (len(row) + 1)
                unit   = avail / total
                x      = kb_x + self.KEY_PAD_H
                for label in row:
                    kw = int(unit * BOTTOM_WIDTHS.get(label, 1.0))
                    self._keys.append(Key(label, x, y, kw, key_h, idx))
                    x   += kw + self.KEY_PAD_H
                    idx += 1

    def get_key_regions(self) -> List[Key]:
        return self._keys

    # ── State ──────────────────────────────────────────────────────────────────
    def toggle_caps(self):
        self.caps_lock = not self.caps_lock

    def toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.theme      = THEMES[self.theme_name]

    def trigger_key_press(self, key_index: Optional[int]):
        if key_index is not None and 0 <= key_index < len(self._keys):
            self._keys[key_index].press_t = time.time()

    # ── Main render ────────────────────────────────────────────────────────────
    def render(self, frame, hand_data, typed_text,
               fps, gesture_path, show_help) -> np.ndarray:
        canvas = frame.copy()

        # Draw hand skeleton (all 21 points, subtle)
        if self.cfg.show_landmarks and hand_data.detected and hand_data.landmarks_px:
            from hand_tracking import HAND_CONNECTIONS
            pts = hand_data.landmarks_px
            for s, e in HAND_CONNECTIONS:
                cv2.line(canvas, pts[s], pts[e], (60, 180, 60), 1, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                # Only highlight index tip (8) and thumb tip (4) brightly
                if i in (4, 8):
                    cv2.circle(canvas, pt, 6, (0, 230, 255), -1, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, pt, 3, (40, 140, 40), -1, cv2.LINE_AA)

        self._draw_text_bar(canvas, typed_text)
        self._draw_keyboard(canvas, hand_data)

        if hand_data.detected and hand_data.index_tip:
            self._draw_cursor(canvas, hand_data)
            self._draw_pinch_meter(canvas, hand_data)

        self._draw_hud(canvas, fps, hand_data)

        if show_help:
            self._draw_help(canvas)

        return canvas

    # ── Text bar ───────────────────────────────────────────────────────────────
    def _draw_text_bar(self, canvas, text):
        bx = self.kb_rect[0]
        bw = self.kb_rect[2]
        by = max(4, self.kb_rect[1] - self.TEXT_BAR_H - 6)
        self._glass_rect(canvas, bx, by, bw, self.TEXT_BAR_H,
                         self.theme["text_bar_bg"], 12)
        blink   = (time.time() - self._cursor_t) % (2*self.CURSOR_BLINK) < self.CURSOR_BLINK
        display = text[-52:] + ("|" if blink else " ")
        cv2.putText(canvas, "Text:", (bx+14, by+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    tuple(max(0, c-80) for c in self.theme["text_color"]),
                    1, cv2.LINE_AA)
        cv2.putText(canvas, display, (bx+14, by + self.TEXT_BAR_H//2 + 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.80,
                    self.theme["text_color"], 1, cv2.LINE_AA)

    # ── Keyboard ───────────────────────────────────────────────────────────────
    def _draw_keyboard(self, canvas, hand_data):
        now    = time.time()
        ix, iy = (hand_data.index_tip if hand_data.detected
                  and hand_data.index_tip else (-999, -999))

        for k in self._keys:
            on_key    = k.contains(ix, iy)
            press_age = now - k.press_t
            pressed   = press_age < self.ANIM_PRESS_DUR

            # ── Background ────────────────────────────────────────────────────
            if pressed:
                bg  = self.theme["press_bg"]
                bdr = self.theme["press_border"]
                bdt = 3
            elif on_key:
                bg  = self.theme["hover_bg"]
                bdr = self.theme["hover_border"]
                bdt = 3
            elif k.label == "BACK":
                bg  = self.theme["back_bg"]
                bdr = self.theme["key_border"]
                bdt = 1
            elif k.label in ("CLEAR",):
                bg  = self.theme["clear_bg"]
                bdr = self.theme["key_border"]
                bdt = 1
            elif k.is_special:
                bg  = self.theme["special_bg"]
                bdr = self.theme["key_border"]
                bdt = 1
            else:
                bg  = self.theme["key_bg"]
                bdr = self.theme["key_border"]
                bdt = 1

            self._glass_rect(canvas, k.x, k.y, k.w, k.h, bg, self.KEY_RADIUS)
            self._rounded_border(canvas, k.x, k.y, k.w, k.h,
                                 self.KEY_RADIUS, bdr[:3] if len(bdr)==4 else bdr, bdt)

            # Outer glow on hover
            if on_key and not pressed:
                self._draw_glow(canvas, k.x, k.y, k.w, k.h, (0, 200, 255))

            # ── Label ─────────────────────────────────────────────────────────
            raw_label = k.label
            if k.is_special:
                disp = KEY_LABELS.get(raw_label, raw_label)
            else:
                disp = raw_label.upper() if self.caps_lock else raw_label.lower()

            scale = 0.46 if k.is_special else 0.64
            font  = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), _ = cv2.getTextSize(disp, font, scale, 1)
            tx = k.x + (k.w - tw) // 2
            ty = k.y + (k.h + th) // 2
            cv2.putText(canvas, disp, (tx, ty), font, scale,
                        self.theme["key_text"], 1, cv2.LINE_AA)

    # ── Cursor — index tip + thumb tip only ────────────────────────────────────
    def _draw_cursor(self, canvas, hand_data):
        ix, iy   = hand_data.index_tip
        tx, ty   = hand_data.thumb_tip
        dist     = hand_data.pinch_dist
        thresh   = self.cfg.pinch_threshold
        pinching = dist < thresh

        # Colour: cyan when pinching, green when on key, white otherwise
        on_key = any(k.contains(ix, iy) for k in self._keys)
        if pinching:
            col_idx = (0, 255, 220)
            col_thm = (0, 255, 220)
        elif on_key:
            col_idx = (0, 255, 80)
            col_thm = (200, 200, 200)
        else:
            col_idx = (220, 220, 255)
            col_thm = (180, 180, 180)

        # ── Index tip — big ring + dot ────────────────────────────────────────
        cv2.circle(canvas, (ix, iy), 22, col_idx, 2, cv2.LINE_AA)
        cv2.circle(canvas, (ix, iy),  6, col_idx, -1, cv2.LINE_AA)

        # ── Thumb tip — smaller dot ───────────────────────────────────────────
        cv2.circle(canvas, (tx, ty), 10, col_thm, 2, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty),  4, col_thm, -1, cv2.LINE_AA)

        # Line between them
        cv2.line(canvas, (ix, iy), (tx, ty), col_idx, 1, cv2.LINE_AA)

        # Distance label between the two tips
        mx, my = (ix+tx)//2, (iy+ty)//2
        cv2.putText(canvas, f"{dist:.0f}px", (mx+8, my-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col_idx, 1, cv2.LINE_AA)

        # CLICK feedback
        if pinching:
            cv2.circle(canvas, (mx, my), 11, (0, 255, 200), -1, cv2.LINE_AA)
            cv2.putText(canvas, "CLICK", (ix+26, iy-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 200), 2, cv2.LINE_AA)

    # ── Pinch meter ────────────────────────────────────────────────────────────
    def _draw_pinch_meter(self, canvas, hand_data):
        dist   = hand_data.pinch_dist
        thresh = self.cfg.pinch_threshold
        ratio  = max(0.0, min(1.0, 1.0 - dist / (thresh * 2.2)))

        bx, by = 10, self.fh - 36
        bw, bh = 180, 18

        # Background
        cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh),
                      self.theme["meter_empty"], -1)
        cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh),
                      self.theme["meter_border"], 1)

        # Fill
        fw = int(bw * ratio)
        if fw > 0:
            col = self.theme["meter_full"] if ratio > 0.72 else self.theme["meter_fill"]
            cv2.rectangle(canvas, (bx, by), (bx+fw, by+bh), col, -1)

        # Threshold line
        tx = bx + int(bw * 0.62)
        cv2.line(canvas, (tx, by-3), (tx, by+bh+3), (60, 60, 255), 2)

        # Labels
        cv2.putText(canvas, "PINCH METER", (bx, by-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 170, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "CLICK ZONE", (tx+4, by+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 255), 1, cv2.LINE_AA)

    # ── HUD ────────────────────────────────────────────────────────────────────
    def _draw_hud(self, canvas, fps, hand_data):
        cv2.putText(canvas, f"FPS {fps:.0f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    self.theme["fps_color"], 2, cv2.LINE_AA)
        if self.caps_lock:
            cv2.putText(canvas, "CAPS", (10, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 255), 1, cv2.LINE_AA)
        status = "Hand OK" if hand_data.detected else "No Hand"
        color  = (60, 255, 60) if hand_data.detected else (80, 80, 220)
        cv2.putText(canvas, status, (self.fw-110, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

    # ── Help overlay ───────────────────────────────────────────────────────────
    def _draw_help(self, canvas):
        lines = [
            "CONTROLS",
            "─────────────────────",
            "Q / ESC   quit",
            "T         toggle theme",
            "L         landmarks on/off",
            "C         clear text",
            "H         this help",
            "",
            "HOW TO CLICK A KEY",
            "─────────────────────",
            "1. Point index finger at key",
            "   Key glows blue when hovering",
            "2. Bring thumb to index tip",
            "   Watch PINCH METER fill up",
            "3. Past blue line = CLICK fires",
            "4. Open fingers to reset",
            "",
            "TIPS",
            "─────────────────────",
            "Keep hand 40-70cm from cam",
            "Good lighting = better track",
            "Only index + thumb matter",
        ]
        ox, oy = 38, 60
        bh = len(lines) * 20 + 22
        self._glass_rect(canvas, ox-10, oy-18, 318, bh, (12,14,30,225), 14)
        for i, line in enumerate(lines):
            bold = line.startswith("─") or line in ("CONTROLS","HOW TO CLICK A KEY","TIPS")
            cv2.putText(canvas, line, (ox, oy + i*20),
                        cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX,
                        0.48 if bold else 0.43,
                        (70, 200, 255) if bold else (210, 230, 255),
                        1, cv2.LINE_AA)

    # ── Primitives ─────────────────────────────────────────────────────────────
    def _glass_rect(self, canvas, x, y, w, h, color_bgra, radius=8):
        ch, cw = canvas.shape[:2]
        x1, y1 = max(0, x),    max(0, y)
        x2, y2 = min(cw, x+w), min(ch, y+h)
        if x2 <= x1 or y2 <= y1:
            return
        roi  = canvas[y1:y2, x1:x2]
        blur = cv2.GaussianBlur(roi, (self.BLUR_KSIZE, self.BLUR_KSIZE), 0)
        b, g, r, a = color_bgra
        tint = np.full_like(roi, (b, g, r), dtype=np.uint8)
        canvas[y1:y2, x1:x2] = cv2.addWeighted(blur, 1-a/255, tint, a/255, 0)
        self._rounded_border(canvas, x, y, w, h, radius, (b, g, r))

    def _rounded_border(self, img, x, y, w, h, r, color, t=1):
        x2, y2 = x+w, y+h
        cv2.line(img, (x+r,y),   (x2-r,y),   color, t, cv2.LINE_AA)
        cv2.line(img, (x+r,y2),  (x2-r,y2),  color, t, cv2.LINE_AA)
        cv2.line(img, (x,y+r),   (x,y2-r),   color, t, cv2.LINE_AA)
        cv2.line(img, (x2,y+r),  (x2,y2-r),  color, t, cv2.LINE_AA)
        cv2.ellipse(img,(x+r, y+r), (r,r),180,0,90,color,t,cv2.LINE_AA)
        cv2.ellipse(img,(x2-r,y+r), (r,r),270,0,90,color,t,cv2.LINE_AA)
        cv2.ellipse(img,(x+r, y2-r),(r,r), 90,0,90,color,t,cv2.LINE_AA)
        cv2.ellipse(img,(x2-r,y2-r),(r,r),  0,0,90,color,t,cv2.LINE_AA)

    def _draw_glow(self, img, x, y, w, h, color):
        p = 7
        ov = img.copy()
        self._rounded_border(ov, x-p, y-p, w+2*p, h+2*p,
                              self.KEY_RADIUS+p, color, 3)
        cv2.addWeighted(ov, 0.50, img, 0.50, 0, img)