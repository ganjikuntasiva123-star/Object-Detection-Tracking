"""
Object detector wrapper using Ultralytics YOLO
Optimized for Render Free Hosting
"""

import numpy as np
from ultralytics import YOLO


COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train',
    'truck','boat','traffic light','fire hydrant','stop sign',
    'parking meter','bench','bird','cat','dog','horse','sheep',
    'cow','elephant','bear','zebra','giraffe','backpack','umbrella',
    'handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports ball','kite','baseball bat','baseball glove',
    'skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli',
    'carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet',
    'tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator',
    'book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]


class YOLODetector:

    def __init__(
        self,
        model_name="yolov8n.pt",
        conf_threshold=0.4,
        iou_threshold=0.45,
        device="cpu",
        classes=None
    ):

        self.model=None

        self.model_name="yolov8n.pt"

        self.conf_threshold=conf_threshold

        self.iou_threshold=iou_threshold

        self.device=device

        self.filter_classes=classes

        self.class_names=COCO_CLASSES


    def load_model(self):

        if self.model is None:

            print("Loading model...")

            self.model=YOLO(self.model_name)

            print("Model Loaded")


    def detect(self,image):

        self.load_model()

        results=self.model(

            image,

            conf=self.conf_threshold,

            iou=self.iou_threshold,

            device=self.device,

            verbose=False,

            imgsz=320

        )

        detections=[]

        if len(results)==0:

            return detections

        result=results[0]

        if result.boxes is None:

            return detections

        boxes=result.boxes.xyxy.cpu().numpy()

        confs=result.boxes.conf.cpu().numpy()

        classes=result.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):

            class_id=classes[i]

            if self.filter_classes:

                if class_id not in self.filter_classes:

                    continue

            detections.append([

                float(boxes[i][0]),

                float(boxes[i][1]),

                float(boxes[i][2]),

                float(boxes[i][3]),

                float(confs[i]),

                int(class_id)

            ])

        return detections


    def get_class_name(self,class_id):

        if 0<=class_id<len(self.class_names):

            return self.class_names[class_id]

        return str(class_id)
