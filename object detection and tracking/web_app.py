import os
import cv2
import time
import threading

from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename

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

        self.frame=None

        self.fps=0

        self.frames=0


state=AppState()


def worker():

    previous=time.time()

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

        state.fps=1/max(

            now-previous,

            0.001

        )

        previous=now

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

                    b'Content-Type: image/jpeg\r\n\r\n'

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


@app.route("/api/upload",methods=["POST"])
def upload():

    if "file" not in request.files:

        return jsonify({

            "status":"error",

            "message":"No file"

        }),400

    file=request.files["file"]

    filename=secure_filename(

        file.filename

    )

    filepath=os.path.join(

        UPLOAD_FOLDER,

        filename

    )

    file.save(

        filepath

    )

    return jsonify({

        "status":"success",

        "path":filepath

    })


@app.route("/api/start",methods=["POST"])
def start():

    if state.running:

        return jsonify({

            "status":"already_running"

        })

    data=request.get_json()

    if not data:

        return jsonify({

            "status":"error",

            "message":"Missing data"

        }),400

    source=data.get(

        "source"

    )

    if not source:

        return jsonify({

            "status":"error",

            "message":"No source"

        }),400

    cap=cv2.VideoCapture(

        source

    )

    if not cap.isOpened():

        return jsonify({

            "status":"error",

            "message":"Cannot open video"

        }),400

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
