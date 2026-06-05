import os
import cv2
import time
import threading

from flask import Flask, render_template, Response, jsonify, request

from detector import YOLODetector, COCO_CLASSES
from tracker import SORTTracker

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class AppState:

    def __init__(self):

        self.running = False
        self.cap = None
        self.thread = None

        self.current_frame = None

        self.detector = None
        self.tracker = None

        self.tracked_objects = []

        self.frame_count = 0
        self.fps = 0

        self.event_log = []


state = AppState()


def video_worker():

    previous = time.time()

    while state.running:

        if state.cap is None:

            break

        ret, frame = state.cap.read()

        if not ret:

            state.running = False
            break

        detections = []

        try:

            detections = state.detector.detect(frame)

        except:

            detections = []

        try:

            tracked = state.tracker.update(detections)

        except:

            tracked = []

        state.tracked_objects = tracked

        for obj in tracked:

            try:

                x1, y1, x2, y2, track_id, class_id, conf = obj

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0,255,0),
                    2
                )

                label = f"{track_id}"

                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    2
                )

            except:

                pass

        now = time.time()

        if now != previous:

            state.fps = 1/(now-previous)

        previous = now

        state.frame_count += 1

        ret, buffer = cv2.imencode(".jpg", frame)

        if ret:

            state.current_frame = buffer.tobytes()

    state.running = False

    if state.cap:

        state.cap.release()


@app.route("/")
def index():

    return render_template("index.html")


@app.route("/video_feed")
def video_feed():

    def generate():

        while True:

            if state.current_frame:

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + state.current_frame +
                    b'\r\n'
                )

            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/start", methods=["POST"])
def api_start():

    if state.running:

        return jsonify({

            "status":"already_running"

        })

    data = request.get_json(silent=True) or {}

    source = str(data.get("source","0"))

    model = data.get(

        "model",

        "yolo11n.pt"

    )

    try:

        state.detector = YOLODetector(

            model_name=model,

            device="cpu"

        )

        state.tracker = SORTTracker()

    except Exception as e:

        return jsonify({

            "status":"error",

            "message":str(e)

        }),500

    # FIX WEB CAM CRASH ON RENDER

    if source == "0":

        return jsonify({

            "status":"error",

            "message":"Webcam only works locally. Use Video File on Render."

        }),400

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():

        return jsonify({

            "status":"error",

            "message":"Cannot open source"

        }),500

    state.cap = cap

    state.running = True

    state.frame_count = 0

    state.fps = 0

    thread = threading.Thread(

        target=video_worker,

        daemon=True

    )

    thread.start()

    state.thread = thread

    return jsonify({

        "status":"started"

    })


@app.route("/api/stop", methods=["POST"])
def stop():

    state.running = False

    if state.cap:

        state.cap.release()

    return jsonify({

        "status":"stopped"

    })


@app.route("/api/status")
def status():

    return jsonify({

        "running": state.running,

        "fps": round(state.fps,1),

        "tracks": len(state.tracked_objects),

        "frames": state.frame_count

    })


@app.route("/api/classes")
def classes():

    return jsonify(COCO_CLASSES)


if __name__ == "__main__":

    port = int(

        os.environ.get(

            "PORT",

            5000

        )

    )

    app.run(

        host="0.0.0.0",

        port=port,

        threaded=True

    )
