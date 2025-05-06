#detector.py
import cv2
from ultralytics import YOLO

def init_detector(model_path: str = "yolov5nu.pt") -> YOLO:
    # Initialize YOLO model
    model = YOLO(model_path)
    model.to("cpu")   # force CPU for low-end GPU
    return model

def detect_people(model: YOLO, frame: any, conf_thresh: float = 0.4):
    # Resize for balanced performance
    resized = cv2.resize(frame, (640, 480))
    results = model(resized)
    boxes, scores = [], []
    det = results[0].boxes
    if det is not None and det.xyxy is not None:
        for box, conf, cls in zip(det.xyxy.cpu().numpy(),
                                  det.conf.cpu().numpy(),
                                  det.cls.cpu().numpy()):
            if int(cls) == 0 and conf >= conf_thresh:
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))
                scores.append(float(conf))
    return boxes, scores, resized
