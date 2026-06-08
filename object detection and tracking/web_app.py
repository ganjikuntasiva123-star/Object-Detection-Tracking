import os
import cv2
import threading
import time

from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

class AppState:
    def __init__(self):
        self.running=False
        self.cap=None
        self.thread=None
        self.frame=None
        self.fps=0
        self.frames=0

state=AppState()


def worker():

    prev=time.time()

    while state.running:

        if state.cap is None:
            break

        ret,frame=state.cap.read()

        if not ret:
            state.running=False
            break

        cv2.putText(
            frame,
            "Object Detection Demo",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        now=time.time()

        state.fps=1/max(now-prev,0.001)

        prev=now

        state.frames+=1

        _,buffer=cv2.imencode(
            ".jpg",
            frame
        )

        state.frame=buffer.tobytes()


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/video_feed")
def video_feed():

    def generate():

        while True:

            if state.frame:

                yield(
                    b'--frame\r\n'
                    b'Content-Type:image/jpeg\r\n\r\n'
                    +state.frame+
                    b'\r\n'
                )

            time.sleep(
                0.03
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/start",methods=["POST"])
def start():

    if state.running:

        return jsonify(
            {"status":"already_running"}
        )

    data=request.get_json() or {}

    source=data.get(
        "source",
        ""
    )

    # IMPORTANT FIX
    # DO NOT USE WEBCAM ON RENDER

    if source=="0":

        return jsonify({
            "status":"error",
            "message":"Render cannot access your webcam. Upload video file."
        }),400

 def start():

    source="video.mp4"

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        return

    state.cap=cap
    state.running=True

    thread=threading.Thread(
        target=worker,
        daemon=True
    )

    thread.start()

    state.thread=thread

    return jsonify({
        "status":"started"
    })


@app.route("/api/stop",methods=["POST"])
def stop():

    state.running=False

    if state.cap:

        state.cap.release()

    return jsonify({
        "status":"stopped"
    })


@app.route("/api/status")
def status():

    return jsonify({

        "running":state.running,

        "fps":round(
            state.fps,
            1
        ),

        "frames":state.frames

    })


if __name__=="__main__":

    port=int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port

    )
