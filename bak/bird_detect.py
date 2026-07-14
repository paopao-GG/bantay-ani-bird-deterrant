#!/usr/bin/env python3
import os, time, cv2
from datetime import datetime
from ultralytics import YOLO
from picamera2 import Picamera2

# --- Config ---
OUT_DIR = "bird_snaps"
os.makedirs(OUT_DIR, exist_ok=True)
CONF_THRES = 0.4
IMG_SIZE = 416
FRAME_SKIP = 2
PREVIEW = True

# --- Model ---
model = YOLO("yolov8n.pt")
bird_label = "bird"

# --- Camera ---
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (1280, 720)},
    controls={"FrameDurationLimits": (33333, 33333)},  # target ~30fps
)
picam2.configure(config)
picam2.start()

frame_count = 0
last_save = 0

print("Press Q to quit")
try:
    while True:
        frame = picam2.capture_array()
        frame_count += 1

        run_detect = (frame_count % FRAME_SKIP == 0)
        if run_detect:
            results = model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                verbose=False
            )
            for r in results:
                for b in r.boxes:
                    cls = int(b.cls[0])
                    label = r.names.get(cls, str(cls))
                    conf = float(b.conf[0])
                    if label == bird_label:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}",
                                    (x1, max(20, y1-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0), 2)
                        if time.time() - last_save > 3:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            cv2.imwrite(os.path.join(OUT_DIR, f"{ts}.jpg"), frame)
                            last_save = time.time()

        if PREVIEW:
            cv2.imshow("Bird Detection", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
