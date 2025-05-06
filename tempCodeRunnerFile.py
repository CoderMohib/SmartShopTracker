import cv2
from detector import init_detector, detect_people
from tracker import init_tracker, update_tracks
from recognizer import recognize_face, known_embeddings, known_names
from age_estimator import estimate_age
from database import get_customer_by_name, add_customer, update_customer
from utils import save_image, get_timestamp

# seconds of absence before counting a return visit
SESSION_TIMEOUT = 10

# map Deep SORT track IDs → assigned person name
track_name_map = {}

def main():
    model = init_detector()
    tracker = init_tracker()
    cap = cv2.VideoCapture(1)  # external camera on index 1
    print("[INFO] Starting SmartShop Tracker. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Detect & 2) Track
        boxes, scores, resized = detect_people(model, frame)
        tracks = update_tracks(tracker, boxes, scores, resized)

        for track in tracks:
            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            # safe bounding-box crop
            h, w, _ = resized.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = resized[y1:y2, x1:x2]

            # 3) Face recognition
            name, emb, face_found = recognize_face(crop)

            if tid in track_name_map:
                # reuse existing identity for this track
                name = track_name_map[tid]
            else:
                # first valid face for this track
                if face_found and emb is not None:
                    # if no match, create new name
                    if name is None:
                        name = f"Person_{len(known_names) + 1}"
                    # register globally
                    known_names.append(name)
                    known_embeddings.append(emb)
                    track_name_map[tid] = name
                else:
                    # no face yet: skip until we get one
                    continue

            # 4) Age estimation
            age = estimate_age(crop)

            # 5) Database insert/update
            now = get_timestamp()
            customer = get_customer_by_name(name)
            if not customer:
                # first-ever visit
                data = {
                    "name": name,
                    "track_id": tid,
                    "first_seen": now,
                    "last_seen": now,
                    "visit_count": 1,
                    "total_time": 0,
                    "age": age,
                    "image_path": save_image(crop, name)
                }
                add_customer(data)
            else:
                elapsed = (now - customer["last_seen"]).total_seconds()
                visits = customer["visit_count"] + 1 if elapsed > SESSION_TIMEOUT else customer["visit_count"]
                total_time = customer["total_time"] + elapsed
                update_customer(name, {
                    "last_seen": now,
                    "visit_count": visits,
                    "total_time": total_time,
                    "age": age
                })

            # 6) Draw bounding box & label
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} | Age: {age}"
            cv2.putText(resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("SmartShop Tracker", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
