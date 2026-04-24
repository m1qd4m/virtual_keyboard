"""
AI Virtual Keyboard — main.py
Run:  python main.py
      python main.py --width 960 --height 540
      python main.py --cam 1
      python main.py --no-sound

Controls (keyboard shortcuts while window is focused):
  Q / ESC   quit
  T         toggle dark/light theme
  L         toggle landmark overlay
  C         clear typed text
  H         toggle help overlay
  S         toggle sound
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2, argparse, threading, time

from config            import Config
from fps_counter       import FPSCounter
from audio             import AudioFeedback
from hand_tracking     import HandTracker
from keyboard_ui       import KeyboardUI
from gesture_logic     import GestureEngine
from prediction_engine import PredictionEngine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--width",    type=int, default=1280)
    p.add_argument("--height",   type=int, default=720)
    p.add_argument("--cam",      type=int, default=0)
    p.add_argument("--fps",      type=int, default=30)
    p.add_argument("--no-sound", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config(args)

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(
        args.cam,
        cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        print("[ERROR] Camera not found. Try --cam 1")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera {W}x{H} — press H in the window for help")

    tracker   = HandTracker(cfg)
    ui        = KeyboardUI(W, H, cfg)
    gesture   = GestureEngine(cfg)
    predictor = PredictionEngine()   # kept but not shown (no suggestion bar)
    fps_ctr   = FPSCounter(smoothing=20)
    audio     = AudioFeedback(enabled=cfg.sound_enabled)

    typed_text = ""
    show_help  = False

    # ── Background frame reader thread ────────────────────────────────────────
    frame_lock   = threading.Lock()
    latest_frame = [None]
    stop_event   = threading.Event()

    def reader():
        while not stop_event.is_set():
            ok, f = cap.read()
            if ok:
                with frame_lock:
                    latest_frame[0] = f

    rt = threading.Thread(target=reader, daemon=True)
    rt.start()
    time.sleep(0.25)

    cv2.namedWindow("AI Virtual Keyboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Virtual Keyboard", W, H)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        with frame_lock:
            raw = latest_frame[0]
        if raw is None:
            continue

        frame = cv2.flip(raw, 1)
        fps_ctr.tick()

        hand_data = tracker.process(frame)
        events    = gesture.update(hand_data, ui.get_key_regions())

        for ev in events:
            if ev["type"] == "key_press":
                key = ev["key"]
                if   key == "BACK":  typed_text = typed_text[:-1]
                elif key == "CLEAR": typed_text = ""
                elif key == "SPACE": typed_text += " "
                elif key == "ENTER": typed_text += "\n"
                elif key == "CAPS":  ui.toggle_caps()
                else:
                    typed_text += key.upper() if ui.caps_lock else key.lower()
                audio.play_click()
                ui.trigger_key_press(ev.get("key_index"))

        canvas = ui.render(
            frame      = frame,
            hand_data  = hand_data,
            typed_text = typed_text,
            fps        = fps_ctr.get(),
            gesture_path = gesture.get_swipe_path(),
            show_help  = show_help,
        )
        cv2.imshow("AI Virtual Keyboard", canvas)

        k = cv2.waitKey(1) & 0xFF
        if   k in (ord("q"), 27): break
        elif k == ord("t"): ui.toggle_theme()
        elif k == ord("l"): cfg.show_landmarks = not cfg.show_landmarks
        elif k == ord("s"): audio.toggle()
        elif k == ord("c"): typed_text = ""
        elif k == ord("h"): show_help  = not show_help

    stop_event.set()
    rt.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("[INFO] Closed.")


if __name__ == "__main__":
    main()