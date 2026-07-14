#!/usr/bin/env python3
import os, time, re, cv2
from datetime import datetime
from ultralytics import YOLO
from picamera2 import Picamera2

# ===== Relays =====
import RPi.GPIO as GPIO

# ---------- Config ----------
OUT_DIR = "bird_snaps"; os.makedirs(OUT_DIR, exist_ok=True)
PREVIEW = True

# Detection tuning for small/distant birds
MODEL_PATH = "best.onnx"          # <--- your ONNX path
CONF_THRES = 0.10                 # lower for recall
IOU_THRES  = 0.45
PREFERRED_SIZES = [512, 640]      # try 512 first; will auto-learn fixed size (e.g., 640)
FRAME_SKIP = 2                    # run detector every N frames
ROI_FRACTION_TOP = 0.60           # analyze top 60% (sky/field). Set to 1.0 to use full frame
H_MIN = 28                        # minimum box height in pixels to accept
PERSIST_FRAMES = 2                # require detections across N detection passes

SAVE_COOLDOWN = 3.0               # seconds between saved frames

# Camera / exposure for motion
CAP_W, CAP_H = 1280, 720
TARGET_FPS = 30
FAST_SHUTTER_US = 1000
ANALOG_GAIN = 4.0

# Relays
RELAY_PINS = [17, 27, 22, 23]
RELAY_ACTIVE_LOW = True
BUZZ_ON_SECS = 2.0
BUZZ_COOLDOWN = 2.0

# ---------- Model ----------
model = YOLO(MODEL_PATH)
BIRD_LABEL = "bird"

def get_names(m):
    names = getattr(getattr(m, "model", m), "names", None)
    return names if names else {}

NAMES = get_names(model)
BIRD_ID = None
if isinstance(NAMES, dict):
    for k, v in NAMES.items():
        if v == BIRD_LABEL:
            BIRD_ID = int(k)

# ---------- Camera ----------
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (CAP_W, CAP_H)},
    controls={"FrameDurationLimits": (int(1e6/TARGET_FPS), int(1e6/TARGET_FPS))}
)
picam2.configure(config)
picam2.start()
picam2.set_controls({
    "ExposureTime": FAST_SHUTTER_US,
    "AnalogueGain": ANALOG_GAIN,
})

# ---------- Relays ----------
def _on():  return GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH
def _off(): return GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
def relays_setup():
    GPIO.setmode(GPIO.BCM)
    for p in RELAY_PINS: GPIO.setup(p, GPIO.OUT, initial=_off())
def relays_on():
    for p in RELAY_PINS: GPIO.output(p, _on())
def relays_off():
    for p in RELAY_PINS: GPIO.output(p, _off())

# ---------- Helpers ----------
def save_snap(img, conf):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(OUT_DIR, f"bird_{ts}_{int(conf*100)}.jpg"), img)

_detect_fixed_size = None

def _extract_expected_size_from_err(msg: str):
    # Looks for "Expected: 640" in onnxruntime error text
    m = re.search(r"Expected:\s*(\d+)", msg)
    return int(m.group(1)) if m else None

def _predict_with_size(frame_roi, size):
    kwargs = dict(source=frame_roi, imgsz=size, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
    if BIRD_ID is not None:
        kwargs["classes"] = [BIRD_ID]
    return model.predict(**kwargs)[0]

def run_multiscale(frame_roi):
    """
    Try preferred sizes. If ONNX throws a fixed-size error (e.g., expects 640),
    learn it once and reuse that size thereafter.
    """
    global _detect_fixed_size

    sizes_to_try = [_detect_fixed_size] if _detect_fixed_size else PREFERRED_SIZES
    sizes_to_try = [s for s in sizes_to_try if s]

    for s in sizes_to_try:
        try:
            r = _predict_with_size(frame_roi, s)
            if _detect_fixed_size is None:
                _detect_fixed_size = s
                print(f"[detector] Using imgsz={_detect_fixed_size}")
            return r
        except RuntimeError as e:
            exp = _extract_expected_size_from_err(str(e))
            if exp:
                _detect_fixed_size = exp
                print(f"[detector] ONNX expects fixed imgsz={exp}, retrying…")
                r = _predict_with_size(frame_roi, exp)
                return r
            else:
                raise
    return None

# track persistence across detection passes
persist_counter = 0
frame_idx = 0
last_save = 0.0
last_buzz = 0.0
buzz_until = 0.0

print("Running… press Q to quit")
relays_setup()

try:
    while True:
        frame = picam2.capture_array()
        frame_idx += 1

        H, W = frame.shape[:2]
        roi_h = int(H * ROI_FRACTION_TOP)
        roi = frame[:roi_h, :, :]

        vis = frame.copy()
        bird_seen_this_pass = False
        best_conf = 0.0

        if frame_idx % FRAME_SKIP == 0:
            r = run_multiscale(roi)
            if r and r.boxes is not None and len(r.boxes):
                names = getattr(r, "names", None) or NAMES
                for b in r.boxes:
                    cls = int(b.cls[0]) if getattr(b, "cls", None) is not None else -1
                    label = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
                    conf = float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    # Map back to full-frame (ROI is from top, YOFF=0)
                    hpx = (y2 - y1)
                    if label == BIRD_LABEL and hpx >= H_MIN:
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(vis, f"{label} {conf:.2f} h={hpx}px", (x1, max(20, y1-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        bird_seen_this_pass = True
                        best_conf = max(best_conf, conf)

        # persistence filter
        if bird_seen_this_pass:
            persist_counter = min(persist_counter + 1, PERSIST_FRAMES)
        else:
            persist_counter = max(persist_counter - 1, 0)

        stable_detection = (persist_counter >= PERSIST_FRAMES)

        # HUD
        cv2.line(vis, (0, roi_h), (W, roi_h), (200,200,200), 1)
        hud = f"imgsz={_detect_fixed_size or PREFERRED_SIZES} skip={FRAME_SKIP} ROI_top={int(ROI_FRACTION_TOP*100)}% Hmin={H_MIN}px pers={persist_counter}/{PERSIST_FRAMES}"
        cv2.putText(vis, hud, (8, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 2)

        # Snapshot + buzz (rate-limited)
        now_wall = time.time()
        if stable_detection and (now_wall - last_save) > SAVE_COOLDOWN:
            save_snap(frame, best_conf)
            last_save = now_wall
            cv2.putText(vis, "Saved snapshot", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        now = time.monotonic()
        if stable_detection and (now - last_buzz) > BUZZ_COOLDOWN:
            last_buzz = now
            buzz_until = max(buzz_until, now + BUZZ_ON_SECS)

        if now < buzz_until:
            relays_on()
            cv2.putText(vis, "RELAY: ON", (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            relays_off()
            cv2.putText(vis, "RELAY: OFF", (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

        if PREVIEW:
            cv2.imshow("Bird Detection (Custom ONNX + small-object tuning)", vis)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

finally:
    try: picam2.stop()
    except Exception: pass
    relays_off()
    GPIO.cleanup()
    cv2.destroyAllWindows()
