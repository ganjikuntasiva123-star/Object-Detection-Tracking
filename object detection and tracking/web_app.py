import os
import cv2
import time
import threading

from flask import Flask, render_template, jsonify, request, Response

from detector import YOLODetector, COCO_CLASSES
from tracker import SORTTracker

app = Flask(__name__)

UPLOAD_FOLDER="uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

class AppState:

    def __init__(self):

        self.running=False

        self.cap=None

        self.thread=None

        self.current_frame=None

        self.detector=None

        self.tracker=None

        self.frame_count=0

        self.fps=0

        self.tracks=[]

        self.events=[]

state=AppState()


def add_event(msg):

    state.events.append({

        "time":time.strftime("%H:%M:%S"),

        "msg":msg

    })

    if len(state.events)>100:

        state.events=state.events[-100:]


def video_worker():

    prev=time.time()

    while state.running:

        ret,frame=state.cap.read()

        if not ret:

            break

        detections=[]

        try:

            detections=state.detector.detect(
                frame
            )

        except:

            detections=[]

        try:

            tracks=state.tracker.update(
                detections
            )

        except:

            tracks=[]

        state.tracks=tracks

        for t in tracks:

            try:

                x1,y1,x2,y2,track_id,cls,conf=t

                cv2.rectangle(
                    frame,
                    (int(x1),int(y1)),
                    (int(x2),int(y2)),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    str(track_id),
                    (int(x1),int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    2
                )

            except:

                pass

        now=time.time()

        if now-prev>0:

            state.fps=1/(now-prev)

        prev=now

        state.frame_count+=1

        ok,buffer=cv2.imencode(
            ".jpg",
            frame
        )

        if ok:

            state.current_frame=buffer.tobytes()

    state.running=False

    if state.cap:

        state.cap.release()


@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/video_feed")
def video_feed():

    def gen():

        while True:

            if state.current_frame:

                yield(

                    b"--frame\r\n"

                    b"Content-Type:image/jpeg\r\n\r\n"

                    +

                    state.current_frame

                    +

                    b"\r\n"

                )

            time.sleep(
                0.03
            )

    return Response(

        gen(),

        mimetype=

        "multipart/x-mixed-replace; boundary=frame"

    )


@app.route(
    "/api/start",
    methods=["POST"]
)
def start():

    if state.running:

        return jsonify({

            "status":"already_running"

        })

    data=request.get_json(
        silent=True
    ) or {}

    source=data.get(
        "source",
        "0"
    )

    model=data.get(
        "model",
        "yolo11n.pt"
    )

    # webcam disabled on Render

    if source=="0":

        return jsonify({

            "status":"error",

            "message":

            "Webcam works only locally"

        }),400

    try:

        state.detector=

        YOLODetector(

            model_name=model,

            device="cpu"

        )

        state.tracker=

        SORTTracker()

    except Exception as e:

        return jsonify({

            "status":"error",

            "message":str(e)

        }),500

    cap=cv2.VideoCapture(
        source
    )

    if not cap.isOpened():

        return jsonify({

            "status":"error",

            "message":"Cannot open video"

        }),500

    state.cap=cap

    state.running=True

    state.frame_count=0

    state.fps=0

    thread=threading.Thread(

        target=video_worker,

        daemon=True

    )

    thread.start()

    state.thread=thread

    add_event(
        "stream started"
    )

    return jsonify({

        "status":"started"

    })


@app.route(
    "/api/stop",
    methods=["POST"]
)
def stop():

    state.running=False

    add_event(
        "stream stopped"
    )

    return jsonify({

        "status":"stopped"

    })


@app.route("/api/status")
def status():

    return jsonify({

        "running":

        state.running,

        "fps":

        round(

            state.fps,

            1

        ),

        "tracks":

        len(

            state.tracks

        ),

        "frames":

        state.frame_count,

        "detections":

        len(

            state.tracks

        )

    })


@app.route("/api/tracks")
def tracks():

    return jsonify(

        state.tracks

    )


@app.route("/api/events")
def events():

    return jsonify(

        state.events

    )


@app.route("/api/classes")
def classes():

    return jsonify(

        COCO_CLASSES

    )


if __name__=="__main__":

    port=int(

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
