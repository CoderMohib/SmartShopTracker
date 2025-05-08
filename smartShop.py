import cv2
import threading
from detector import init_detector, detect_people
from tracker import init_tracker, update_tracks
from recognizer import recognize_face, known_embeddings, known_names
from analyzer import analyze_face
from database import get_customer_by_name, add_customer, update_customer
from utils import save_image, get_timestamp

SESSION_TIMEOUT = 10

track_name_map = {}
track_age_map = {}
track_gender_map = {}
analyzing_tracks = set()

def analyze_in_background(crop, name, tid):
    """
    Function to run age and gender analysis in a background thread.
    """
    result = analyze_face(crop, name)
    min_age = result["min_age"]
    max_age = result["max_age"]

    gender = result["gender"]

    # Calculate min_age and max_age


    track_age_map[tid] = (min_age, max_age)
    track_gender_map[tid] = gender

    now = get_timestamp()
    customer = get_customer_by_name(name)
    
    if not customer:
        # If customer is new, add to the database
        data = {
            "name": name,
            "track_id": tid,
            "first_seen": now,
            "last_seen": now,
            "visit_count": 1,
            "total_time": 0,
            "min_age": min_age,
            "max_age": max_age,
            "gender": gender,
            "image_path": save_image(crop, name)
        }
        add_customer(data)
    else:
        # Update existing customer with new data
        elapsed = (now - customer["last_seen"]).total_seconds()
        visits = customer["visit_count"] + 1 if elapsed > SESSION_TIMEOUT else customer["visit_count"]
        total_time = customer["total_time"] + elapsed
        min_age = min(customer.get("min_age", min_age), min_age)
        max_age = max(customer.get("max_age", max_age), max_age)
        update_customer(name, {
            "last_seen": now,
            "visit_count": visits,
            "total_time": total_time,
            "min_age": min_age,
            "max_age": max_age,
            "gender": gender
        })

    # Mark this track as analyzed
    analyzing_tracks.discard(tid)

def smartShop(stop_event=None):
    """
    Main function to run the SmartShop Tracker.
    """
    model = init_detector()
    tracker = init_tracker()
    cap = cv2.VideoCapture(1)  # External camera on index 1

    print("[INFO] Starting SmartShop Tracker. Press 'q' to exit.")

    while True:
        if stop_event and stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores, resized = detect_people(model, frame)
        tracks = update_tracks(tracker, boxes, scores, resized)

        for track in tracks:
            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            h, w, _ = resized.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = resized[y1:y2, x1:x2]
            name, emb, face_found = recognize_face(crop)

            if tid in track_name_map:
                name = track_name_map[tid]
            else:
                if face_found and emb is not None:
                    if name is None:
                        name = f"Person_{len(known_names) + 1}"
                    known_names.append(name)
                    known_embeddings.append(emb)
                    track_name_map[tid] = name
                else:
                    continue

            # Run background analysis if not already done
            if tid not in track_age_map or tid not in track_gender_map:
                if tid not in analyzing_tracks:
                    analyzing_tracks.add(tid)
                    threading.Thread(target=analyze_in_background, args=(crop, name, tid)).start()

            # Retrieve age range and gender, or show "Processing..." if still analyzing
            age_range = track_age_map.get(tid, ("Processing...", "Processing..."))
            gender = track_gender_map.get(tid, "Processing...")

            # Update the label to show name, age range, and gender
            label = f"{name} | Age: {age_range[0]} - {age_range[1]} | Gender: {gender}"
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("SmartShop Tracker", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    smartShop()
