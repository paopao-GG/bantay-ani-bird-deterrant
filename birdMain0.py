#!/usr/bin/env python3
import os, time, cv2, random, subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from picamera2 import Picamera2, Preview

# ===== Relay control (RPi.GPIO) =====
import RPi.GPIO as GPIO

# --- Config ---
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = str(BASE_DIR / "snaps"); os.makedirs(OUT_DIR, exist_ok=True)
CONF_THRES = 0.1
IMG_SIZE = 640          # best.onnx has a fixed 640x640 input; yolov8n (dynamic) accepts it too
FRAME_SKIP = 2
PREVIEW = True

# Relay / buzzer config
RELAY_PINS = [17, 27, 22, 23]     # BCM pins -> relay IN1..IN4
RELAY_ACTIVE_LOW = True           # Most Pi relay boards are active-LOW (LOW = ON)
BUZZ_ON_SECS = 2.0                # Relay ON duration per detection
BUZZ_COOLDOWN = 2.0               # Minimum gap between buzz starts

# Audio config (played through the Pi's built-in audio jack via ffplay)
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_FILES = [str(p) for p in sorted(AUDIO_DIR.glob("*.mp3"))]

# --- Models (ONNX) ---
# Detect with BOTH models; a "bird" from either one triggers.
# Matching is case-insensitive so best.onnx ('Bird') and yolov8n.onnx ('bird') both work.
BIRD_LABEL = "bird"
MODEL_PATHS = [
    str(BASE_DIR / "best.onnx"),      # custom single-class 'Bird' model (runs at its native 640)
    str(BASE_DIR / "yolov8n.onnx"),   # generic COCO model (class 'bird')
]

# Try to find the bird class id (optional speed-up), case-insensitively
def get_bird_id(m):
    names = getattr(m, "names", None) or getattr(getattr(m, "model", m), "names", None)
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).lower() == BIRD_LABEL.lower():
                return int(k)
    elif isinstance(names, (list, tuple)):
        for i, v in enumerate(names):
            if str(v).lower() == BIRD_LABEL.lower():
                return int(i)
    return None

# Load each model once, paired with its bird class id
models = [(m, get_bird_id(m)) for m in (YOLO(p) for p in MODEL_PATHS)]

# --- Live preview overlay (RGBA layer; does NOT touch the capture/detection buffer) ---
PREVIEW_W, PREVIEW_H = 1280, 720          # must match the main-stream size below
latest_dets = []        # [(x1, y1, x2, y2, label, conf), ...] from the last detection
overlay_status = ""     # small status line shown top-left

def update_overlay():
    ov = np.zeros((PREVIEW_H, PREVIEW_W, 4), dtype=np.uint8)  # transparent RGBA
    for (x1, y1, x2, y2, label, conf) in latest_dets:
        cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0, 255), 2)   # RGBA: green, opaque
        cv2.putText(ov, f"{label} {conf:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0, 255), 2)
    if overlay_status:
        cv2.putText(ov, overlay_status, (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0, 255), 2)  # RGBA: yellow
    picam2.set_overlay(ov)

# --- Camera ---
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (1280, 720)},
    controls={"FrameDurationLimits": (33333, 33333)},  # target ~30fps
)
picam2.configure(config)
if PREVIEW:
    # QT (software) preview — supports the RGB888 stream. QTGL/OpenGL does NOT
    # accept RGB888, so it is not used here.
    picam2.start_preview(Preview.QT)
picam2.start()

# --- Relay helpers ---
def _on_level():  return GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH
def _off_level(): return GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW

def relays_setup():
    GPIO.setmode(GPIO.BCM)
    for pin in RELAY_PINS:
        GPIO.setup(pin, GPIO.OUT, initial=_off_level())

def relays_on():
    for pin in RELAY_PINS:
        GPIO.output(pin, _on_level())

def relays_off():
    for pin in RELAY_PINS:
        GPIO.output(pin, _off_level())

# --- Audio helper ---
audio_proc = None
def play_random_audio():
    global audio_proc
    if not AUDIO_FILES:
        return
    if audio_proc is not None and audio_proc.poll() is None:
        return  # a clip is still playing; don't overlap
    audio_proc = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
         random.choice(AUDIO_FILES)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

# --- Utils ---
def save_snap(img):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(OUT_DIR, f"{ts}.jpg"), img)

frame_count = 0
last_save = 0.0
SAVE_COOLDOWN = 3.0   # seconds between saved frames

last_buzz = 0.0
buzz_until = 0.0
prev_status = ""

print("Press Ctrl+C to quit")
relays_setup()

try:
    while True:
        frame = picam2.capture_array()
        frame_count += 1

        run_detect = (frame_count % FRAME_SKIP == 0)
        bird_seen = False
        best_conf = 0.0

        if run_detect:
            dets = []   # boxes for the preview overlay (from all models)
            for m, bird_id in models:
                kwargs = dict(source=frame, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)
                if bird_id is not None:
                    kwargs["classes"] = [bird_id]  # restrict to the bird class if known
                results = m.predict(**kwargs)

                for r in results:
                    if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
                        continue
                    names = getattr(r, "names", None) or getattr(getattr(m, "model", m), "names", {})
                    for b in r.boxes:
                        cls = int(b.cls[0]) if getattr(b, "cls", None) is not None else -1
                        label = names.get(cls, str(cls)) if isinstance(names, dict) else (
                            names[cls] if isinstance(names, (list,tuple)) and 0 <= cls < len(names) else str(cls)
                        )
                        conf = float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0

                        if str(label).lower() == BIRD_LABEL.lower():   # case-insensitive
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            dets.append((x1, y1, x2, y2, label, conf))
                            bird_seen = True
                            if conf > best_conf:
                                best_conf = conf

            latest_dets = dets

        # Snapshots (rate-limited)
        now_wall = time.time()
        if bird_seen and (now_wall - last_save) > SAVE_COOLDOWN:
            save_snap(frame)
            last_save = now_wall

        # Relay/buzzer logic
        now = time.monotonic()
        if bird_seen and (now - last_buzz) > BUZZ_COOLDOWN:
            last_buzz = now
            buzz_until = max(buzz_until, now + BUZZ_ON_SECS)
            play_random_audio()

        if now < buzz_until:
            relays_on()
            overlay_status = "RELAY: ON"
        else:
            relays_off()
            overlay_status = "RELAY: OFF"

        # Refresh the preview overlay only on detection frames or when status flips
        if PREVIEW and (run_detect or overlay_status != prev_status):
            update_overlay()
            prev_status = overlay_status

finally:
    try:
        picam2.stop()
    except Exception:
        pass
    if PREVIEW:
        try:
            picam2.stop_preview()
        except Exception:
            pass
    relays_off()
    GPIO.cleanup()
    if audio_proc is not None and audio_proc.poll() is None:
        audio_proc.terminate()
