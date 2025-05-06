#tracker.py
from deep_sort_realtime.deepsort_tracker import DeepSort

def init_tracker(max_age: int = 1) -> DeepSort:
    """
    max_age=1 makes Deep SORT drop a track if it's missing in the very next frame,
    so bounding boxes vanish immediately when someone leaves view.
    """
    return DeepSort(max_age=max_age)

def update_tracks(tracker: DeepSort, boxes: list, scores: list, frame: any):
    """
    Converts YOLO detections into Deep SORT inputs and returns only active, confirmed tracks.
    """
    inputs = [
        ([x, y, x2-x, y2-y], score, frame[y:y2, x:x2])
        for (x, y, x2, y2), score in zip(boxes, scores)
    ]
    tracks = tracker.update_tracks(inputs, frame=frame)
    return [t for t in tracks if t.is_confirmed()]
