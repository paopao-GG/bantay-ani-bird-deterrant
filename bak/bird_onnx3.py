#!/usr/bin/env python3
# Pi 4 + Camera v2 (Bookworm) — YOLOv8n ONNX Runtime, tuned for flying birds
import os, time, cv2
from datetime import datetime
from ultralytics import YOLO
from picamera2 import Picamera2

# ---------- Config ----------
OUT_DIR = "bird_snaps"; os.makedirs(OUT_DIR, exist_ok=True)
PREVIEW = True

# Camera / exposure
CAP_W, CAP_H = 1280, 720                 # fast binned 16:9 mode
TARGET_FPS = 30
FAST_SHUTTER_US = 1000                   # ~1/1000s to freeze wings
ANALOG_GAIN = 4.0                        # raise if underexposed (2.0–8.0 typical)

# Detection
IMG_SIZE = 416                           # 416..512 (448 good compromise)
CONF_THRES = 0.38                        # lower => more recall
IOU_THRES  = 0.45
FRAME_SKIP = 2                           # run detector every N frames
SAVE_COOLDOWN = 3.0                      # seconds between saved frames
ROI_TOP_FRACTION = 0.55                  # analyze top 55% of frame for flying birds

# ---------- Model (ONNX) ----------
# Ultralytics will route this through onnxruntime if installed.
# Make sure yolov8n.onnx is in the same folder as this script.
model = YOLO("yolov8n.onnx")
# COCO 'bird' class id in Ultralytics' mapping is 14
bird_id = 14

# ---------- Camera ----------
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (CAP_W, CAP_H)},
    controls={"FrameDurationLimits": (int(1e6/TARGET_FPS), int(1e6/TARGET_FPS))}
)
picam2.configure(config)
picam2.start()
picam2.set_controls({
    "ExposureTime": FAST_SHUTTER_US,     # microseconds
    "AnalogueGain": ANALOG_GAIN,
})

def save_snap(img, conf):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(OUT_DIR, f"bird_{ts}_{int(conf*100)}.jpg"), img)

# ---------- Main loop ----------
frame_idx = 0
last_save = 0.0
print("Running (ONNX)… press Q to quit.")
try:
    while True:
        frame = picam2.capture_array()
        frame_idx += 1
        H, W = frame.shape[:2]

        # ROI: upper portion (flying birds)
        roi_h = int(H * ROI_TOP_FRACTION)
        roi = frame[:roi_h, :, :]

        vis = frame.copy()
        bird_found = False
        best_conf = 0.0

        if frame_idx % FRAME_SKIP == 0:
            # Only consider the 'bird' class with a small max_det to reduce CPU
            results = model.predict(
                source=roi,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                iou=IOU_THRES,
                classes=[bird_id],
                max_det=20,
                verbose=False
            )
            r = results[0]
            if r.boxes is not None and len(r.boxes):
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    conf = float(b.conf[0])
                    # Map back to full frame (ROI is top-only)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"bird {conf:.2f}", (x1, max(20, y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    bird_found = True
                    best_conf = max(best_conf, conf)

        # Draw ROI boundary + HUD
        cv2.line(vis, (0, roi_h), (W, roi_h), (200, 200, 200), 1)
        cv2.putText(vis, f"ONNX imgsz={IMG_SIZE} skip={FRAME_SKIP} ROI_top={int(ROI_TOP_FRACTION*100)}%",
                    (8, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

        # Rate-limited snapshot on detection
        now = time.time()
        if bird_found and (now - last_save) > SAVE_COOLDOWN:
            save_snap(frame, best_conf)
            last_save = now
            cv2.putText(vis, "Saved snapshot", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if PREVIEW:
            cv2.imshow("Flying Bird Detector — ONNX (Q to quit)", vis)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

finally:
    try: picam2.stop()
    except Exception: pass
    cv2.destroyAllWindows()
