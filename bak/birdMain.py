#!/home/pi/projects/birddet/.venv/bin/python3
import os, time, cv2
from datetime import datetime
from ultralytics import YOLO
from picamera2 import Picamera2

# ===== Relay control (RPi.GPIO) =====
import RPi.GPIO as GPIO

# --- Config ---
OUT_DIR = "bird_snaps"; os.makedirs(OUT_DIR, exist_ok=True)
CONF_THRES = 0.1
IMG_SIZE = 416
FRAME_SKIP = 2
PREVIEW = True

# Relay / buzzer config
RELAY_PINS = [17, 27, 22, 23]     # BCM pins -> relay IN1..IN4
RELAY_ACTIVE_LOW = True           # Most Pi relay boards are active-LOW (LOW = ON)
BUZZ_ON_SECS = 2.0                # Relay ON duration per detection
BUZZ_COOLDOWN = 2.0               # Minimum gap between buzz starts

# --- Model (ONNX) ---
MODEL_PATH = "yolov8n.onnx"       # <- change to your ONNX file if different
model = YOLO(MODEL_PATH)
BIRD_LABEL = "bird"

# Try to find bird class id (optional speed-up)
def get_bird_id(m):
    names = getattr(getattr(m, "model", m), "names", None)
    if isinstance(names, dict):
        for k, v in names.items():
            if v == BIRD_LABEL:
                return int(k)
    elif isinstance(names, (list, tuple)):
        if BIRD_LABEL in names:
            return int(names.index(BIRD_LABEL))
    return None

BIRD_ID = get_bird_id(model)

# --- Camera ---
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (1280, 720)},
    controls={"FrameDurationLimits": (33333, 33333)},  # target ~30fps
)
picam2.configure(config)
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

# --- Utils ---
def save_snap(img):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(OUT_DIR, f"{ts}.jpg"), img)

frame_count = 0
last_save = 0.0
SAVE_COOLDOWN = 3.0   # seconds between saved frames

last_buzz = 0.0
buzz_until = 0.0

print("Press Q to quit")
relays_setup()

try:
    while True:
        frame = picam2.capture_array()
        frame_count += 1
        vis = frame

        run_detect = (frame_count % FRAME_SKIP == 0)
        bird_seen = False
        best_conf = 0.0

        if run_detect:
            kwargs = dict(source=frame, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)
            if BIRD_ID is not None:
                kwargs["classes"] = [BIRD_ID]  # restrict to 'bird' if we know the id
            results = model.predict(**kwargs)

            for r in results:
                if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
                    continue
                names = getattr(r, "names", None) or getattr(getattr(model, "model", model), "names", {})
                for b in r.boxes:
                    cls = int(b.cls[0]) if getattr(b, "cls", None) is not None else -1
                    label = names.get(cls, str(cls)) if isinstance(names, dict) else (
                        names[cls] if isinstance(names, (list,tuple)) and 0 <= cls < len(names) else str(cls)
                    )
                    conf = float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0

                    if label == BIRD_LABEL:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis, f"{label} {conf:.2f}",
                                    (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        bird_seen = True
                        if conf > best_conf:
                            best_conf = conf

        # Snapshots (rate-limited)
        now_wall = time.time()
        if bird_seen and (now_wall - last_save) > SAVE_COOLDOWN:
            save_snap(frame)
            last_save = now_wall
            cv2.putText(vis, "Saved snapshot", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Relay/buzzer logic
        now = time.monotonic()
        if bird_seen and (now - last_buzz) > BUZZ_COOLDOWN:
            last_buzz = now
            buzz_until = max(buzz_until, now + BUZZ_ON_SECS)

        if now < buzz_until:
            relays_on()
            cv2.putText(vis, "RELAY: ON", (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            relays_off()
            cv2.putText(vis, "RELAY: OFF", (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

        if PREVIEW:
            cv2.imshow("Bird Detection (ONNX + Relays)", vis)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

finally:
    try: picam2.stop()
    except Exception: pass
    relays_off()
    GPIO.cleanup()
    cv2.destroyAllWindows()
