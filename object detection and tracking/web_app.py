"""Web Interface for Object Detection and Tracking System.

Provides a browser-based dashboard with:

Live MJPEG video stream with detection/tracking overlays

Real-time controls (model, confidence, source selection)

Stats panel (FPS, track counts, detection counts)"""

import argparse
import os
import time
import threading
import uuid
import webbrowser
import cv2
import numpy as np

from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename
from detector import YOLODetector, COCO_CLASSES
from tracker import SORTTracker
# ------------------------------------------------------------
# Color palette
# ------------------------------------------------------------
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),(255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),(0, 128, 255), (255, 0, 128), (128, 255, 0), (0, 255, 128),]

def get_color(track_id):return COLORS[track_id % len(COLORS)]

---------------------------------------------------------------------------

# Global state (accessed from both the processing thread and Flask routes)
---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.lock = threading.Lock()
def build_components(self):
    """Create or re-create detector and tracker."""
    try:
        self.detector = YOLODetector(
            model_name=self.model_name,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
            classes=self.filter_classes,
        )
        self.tracker = SORTTracker(
            max_age=self.max_age,
            min_hits=3,
            iou_threshold=self.track_iou,
        )
        return True
    except Exception as e:
        print(f"[ERROR] build_components: {e}")
        return False

state = AppState()app = Flask(name)

Upload configuration

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(file)), "uploads")ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_video_file(filename):"""Check if the uploaded file has an allowed video extension."""ext = os.path.splitext(filename)[1].lower()return ext in ALLOWED_VIDEO_EXTENSIONS

===========================================================================

Video processing thread

===========================================================================

def video_worker():"""Background thread that captures, detects, tracks, and encodes frames."""while state.running:# --- Ensure video capture is open ---with state.lock:if state.cap is None or not state.cap.isOpened():time.sleep(0.1)continue

        ret, frame = state.cap.read()
        if not ret:
            # Reached end of file or camera error → stop
            state.running = False
            print("[INFO] End of video stream.")
            break

        # Resize if needed
        if state.resize != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * state.resize), int(h * state.resize)))

        state.frame_count += 1

    # --- Detection ---
    try:
        detections = state.detector.detect(frame)
    except Exception as e:
        print(f"[ERROR] Detection: {e}")
        detections = []

    # --- Tracking ---
    try:
        tracked_objects = state.tracker.update(detections)
    except Exception as e:
        print(f"[ERROR] Tracking: {e}")
        tracked_objects = []

    # --- Log new track events ---
    with state.lock:
        current_ids = set()
        for obj in tracked_objects:
            track_id = int(obj[4])
            current_ids.add(track_id)
            if track_id not in state.seen_track_ids:
                state.seen_track_ids.add(track_id)
                class_id = int(obj[5])
                cls_name = "unknown"
                if state.detector:
                    cls_name = state.detector.get_class_name(class_id)
                state.event_log.append({
                    "time": time.strftime("%H:%M:%S"),
                    "type": "new_track",
                    "track_id": track_id,
                    "class_name": cls_name,
                    "confidence": round(float(obj[6]), 2),
                })
        # Check for lost tracks
        lost_ids = state.seen_track_ids - current_ids
        for lost_id in lost_ids:
            state.event_log.append({
                "time": time.strftime("%H:%M:%S"),
                "type": "track_lost",
                "track_id": lost_id,
                "class_name": "",
                "confidence": 0.0,
            })
        state.seen_track_ids = current_ids
        # Trim log
        if len(state.event_log) > state.max_events:
            state.event_log = state.event_log[-state.max_events:]

    # --- Annotate frame ---
    display = draw_detections(frame, tracked_objects)
    display = draw_info_panel(display)

    # --- Encode to JPEG ---
    ret_jpeg, jpeg_buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ret_jpeg:
        with state.lock:
            state.current_frame = jpeg_buf.tobytes()
            state.num_tracks = len(tracked_objects)
            state.num_detections = len(detections)
            state.tracked_objects = tracked_objects

    # --- FPS ---
    now = time.time()
    if not hasattr(video_worker, "_last_time"):
        video_worker._last_time = now
    dt = now - video_worker._last_time
    video_worker._last_time = now
    if dt > 0:
        instant_fps = 1.0 / dt
        state.fps_history.append(instant_fps)
        if len(state.fps_history) > 30:
            state.fps_history.pop(0)
        state.fps = sum(state.fps_history) / len(state.fps_history)

===========================================================================

Drawing helpers (reused from main.py)

===========================================================================

def draw_detections(frame, tracked_objects, show_classes=True):display = frame.copy()h, w = display.shape[:2]

for obj in tracked_objects:
    x1, y1, x2, y2, track_id, class_id, confidence = obj
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    color = get_color(track_id)

    # Bounding box
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    # Corner accents
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    if corner_len > 5:
        cv2.line(display, (x1, y1), (x1 + corner_len, y1), color, 2)
        cv2.line(display, (x1, y1), (x1, y1 + corner_len), color, 2)
        cv2.line(display, (x2, y1), (x2 - corner_len, y1), color, 2)
        cv2.line(display, (x2, y1), (x2, y1 + corner_len), color, 2)
        cv2.line(display, (x1, y2), (x1 + corner_len, y2), color, 2)
        cv2.line(display, (x1, y2), (x1, y2 - corner_len), color, 2)
        cv2.line(display, (x2, y2), (x2 - corner_len, y2), color, 2)
        cv2.line(display, (x2, y2), (x2, y2 - corner_len), color, 2)

    # Label
    label_parts = [f"ID:{track_id}"]
    if show_classes and state.detector:
        label_parts.append(state.detector.get_class_name(class_id))
    label = " | ".join(label_parts)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    label_y = y1 - 10 if y1 > 25 else y2 + 25

    cv2.rectangle(display, (x1, label_y - th - 5), (x1 + tw + 10, label_y + 5), color, -1)
    brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    cv2.putText(display, label, (x1 + 5, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

return display

def draw_info_panel(frame):overlay = frame.copy()h, w = frame.shape[:2]cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

with state.lock:
    fps_text = f"{state.fps:.1f}"
    tracks_text = str(state.num_tracks)
    dets_text = str(state.num_detections)
    fcount_text = str(state.frame_count)

info = f"FPS: {fps_text}  |  Tracks: {tracks_text}  |  Detections: {dets_text}  |  Frame: {fcount_text}"
cv2.putText(frame, info, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

legend = "Streaming to web dashboard"
lx = w - 260
cv2.rectangle(frame, (lx, 5), (lx + 250, 35), (0, 0, 0), -1)
cv2.putText(frame, legend, (lx + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

return frame

===========================================================================

Flask routes

===========================================================================

@app.route("/")def index():"""Render the dashboard."""return render_template("index.html")

@app.route("/video_feed")def video_feed():"""MJPEG stream of the annotated video."""def generate():while True:with state.lock:frame_bytes = state.current_frameif frame_bytes is not None:yield (b"--frame\r\n"b"Content-Type: image/jpeg\r\n"b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"+ frame_bytes + b"\r\n")else:# Send a placeholder frameplaceholder = np.zeros((480, 640, 3), dtype=np.uint8)cv2.putText(placeholder, "Waiting for stream...", (160, 240),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)_, buf = cv2.imencode(".jpg", placeholder)fb = buf.tobytes()yield (b"--frame\r\n"b"Content-Type: image/jpeg\r\n"b"Content-Length: " + str(len(fb)).encode() + b"\r\n\r\n"+ fb + b"\r\n")time.sleep(0.03)

return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/api/status')
def api_status():

    return jsonify({

        "online": True,

        "running": getattr(state, "running", False),

        "fps": 0,

        "detections": 0,

        "tracks": len(getattr(state, "tracked_objects", [])),

        "frames": getattr(state, "frame_count", 0)

    })
@app.route("/api/start", methods=["POST"])def api_start():"""Start the video processing pipeline."""# --- Apply settings (fast, under lock) ---with state.lock:if state.running:return jsonify({"status": "already_running"})

    data = request.get_json(silent=True) or {}
    state.source = data.get("source", state.source)
    state.model_name = data.get("model", state.model_name)
    state.conf_threshold = float(data.get("conf", state.conf_threshold))
    state.iou_threshold = float(data.get("iou", state.iou_threshold))
    state.track_iou = float(data.get("track_iou", state.track_iou))
    state.max_age = int(data.get("max_age", state.max_age))
    state.device = data.get("device", state.device)
    filter_cls = data.get("classes", None)
    if filter_cls is not None and isinstance(filter_cls, list):
        state.filter_classes = [int(c) for c in filter_cls]
    else:
        state.filter_classes = None

# --- Build components (slow, outside lock — may download model) ---
if not state.build_components():
    return jsonify({"status": "error", "message": "Failed to load model"}), 500

# --- Open video source ---
with state.lock:
    src = 0 if state.source == "0" else state.source
    if src != 0 and not os.path.exists(src):
        return jsonify({"status": "error", "message": f"Source not found: {src}"}), 400

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return jsonify({"status": "error", "message": "Failed to open video source"}), 500

    state.cap = cap
    state.running = True
    state.frame_count = 0
    state.fps_history = []
    state.event_log = []
    state.seen_track_ids = set()
    video_worker._last_time = time.time()

# Start processing thread
state.thread = threading.Thread(target=video_worker, daemon=True)
state.thread.start()

return jsonify({"status": "started"})

@app.route("/api/stop", methods=["POST"])def api_stop():"""Stop the video processing pipeline."""with state.lock:state.running = Falseif state.cap:state.cap.release()state.cap = Nonestate.current_frame = Nonestate.event_log = []state.seen_track_ids = set()return jsonify({"status": "stopped"})

@app.route("/api/tracks")def api_tracks():"""Return per-track details for the active track list."""with state.lock:tracks = []for obj in state.tracked_objects:x1, y1, x2, y2, track_id, class_id, confidence = objclass_name = "unknown"if state.detector:class_name = state.detector.get_class_name(class_id)tracks.append({"id": int(track_id),"class_id": int(class_id),"class_name": class_name,"confidence": round(float(confidence), 2),})return jsonify(tracks)

@app.route("/api/reset_tracker", methods=["POST"])def api_reset_tracker():"""Reset the SORT tracker."""with state.lock:if state.tracker:state.tracker.reset()state.tracked_objects = []state.seen_track_ids = set()state.event_log.append({"time": time.strftime("%H:%M:%S"),"type": "info","track_id": 0,"class_name": "Tracker reset","confidence": 1.0,})return jsonify({"status": "ok"})

@app.route("/api/update_settings", methods=["POST"])def api_update_settings():"""Update runtime settings (applied on next start)."""data = request.get_json(silent=True) or {}with state.lock:if "conf" in data:state.conf_threshold = float(data["conf"])if "iou" in data:state.iou_threshold = float(data["iou"])if "track_iou" in data:state.track_iou = float(data["track_iou"])if "max_age" in data:state.max_age = int(data["max_age"])return jsonify({"status": "ok"})

@app.route("/api/models")def api_models():"""Return a list of available built-in YOLO models."""models = [{"id": "yolo11n.pt", "name": "YOLO11 Nano"},{"id": "yolo11s.pt", "name": "YOLO11 Small"},{"id": "yolo11m.pt", "name": "YOLO11 Medium"},{"id": "yolov8n.pt", "name": "YOLOv8 Nano"},{"id": "yolov8s.pt", "name": "YOLOv8 Small"},]return jsonify(models)

@app.route("/api/upload", methods=["POST"])def api_upload():"""Upload a video file and return its saved path."""if "file" not in request.files:return jsonify({"status": "error", "message": "No file provided"}), 400

file = request.files["file"]
if file.filename == "":
    return jsonify({"status": "error", "message": "No file selected"}), 400

if not allowed_video_file(file.filename):
    return jsonify({
        "status": "error",
        "message": f"Unsupported video format. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
    }), 400

# Save with a unique filename to prevent overwrites
original_name = secure_filename(file.filename)
unique_name = f"{uuid.uuid4().hex}_{original_name}"
save_path = os.path.join(UPLOAD_FOLDER, unique_name)
file.save(save_path)

file_size = os.path.getsize(save_path)
print(f"[INFO] Uploaded video: {original_name} ({file_size / 1024 / 1024:.1f} MB) -> {save_path}")

return jsonify({
    "status": "ok",
    "file_path": save_path,
    "file_name": original_name,
    "file_size": file_size,
})

@app.route("/api/events")def api_events():"""Return the recent detection event log."""with state.lock:return jsonify(state.event_log[-100:])

@app.route("/api/classes")def api_classes():"""Return COCO class names."""return jsonify(COCO_CLASSES)

# ===========================================================================
# Entry point (Render Compatible)
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Web Dashboard for Object Detection & Tracking"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 5000))
    )

    parser.add_argument(
        "--debug",
        action="store_true"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Initial state
    state.source = args.source
    state.model_name = args.model
    state.conf_threshold = args.conf
    state.device = args.device

    print("=" * 60)
    print("Object Detection & Tracking Dashboard")
    print("=" * 60)

    print("[INFO] Pre-loading model...")
    state.build_components()
    print("[INFO] Ready.")

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", args.port)),
        debug=False,
        threaded=True
    )
