import cv2
from detector import init_detector, detect_people
from tracker import init_tracker, update_tracks
from recognizer import recognize_face, known_embeddings, known_names
from analyzer import analyze_face
from database import get_customer_by_name, add_customer, update_customer
from utils import save_image, get_timestamp

SESSION_TIMEOUT = 10
track_name_map = {}
track_age_map = {}
track_gender_map = {}  # NEW: store gender for each track

def main():
    model = init_detector()
    tracker = init_tracker()
    cap = cv2.VideoCapture(1)  # external camera on index 1
    print("[INFO] Starting SmartShop Tracker. Press 'q' to exit.")

    while True:
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

            # Estimate age/gender only once per track
            if tid not in track_age_map or tid not in track_gender_map:
                result = analyze_face(crop,name)
                age = result["age"]
                gender = result["gender"]
                track_age_map[tid] = age
                track_gender_map[tid] = gender
            else:
                age = track_age_map[tid]
                gender = track_gender_map[tid]
            face_data = analyze_face(crop, name)
            age = face_data["age"]
            gender = face_data["gender"]
            print("i am main gender", gender)
            now = get_timestamp()
            customer = get_customer_by_name(name)
            if not customer:
                data = {
                    "name": name,
                    "track_id": tid,
                    "first_seen": now,
                    "last_seen": now,
                    "visit_count": 1,
                    "total_time": 0,
                    "min_age": age,
                    "max_age": age,
                    "gender": gender,
                    "image_path": save_image(crop, name)
                }
                add_customer(data)
                min_age, max_age = age, age
            else:
                elapsed = (now - customer["last_seen"]).total_seconds()
                visits = customer["visit_count"] + 1 if elapsed > SESSION_TIMEOUT else customer["visit_count"]
                total_time = customer["total_time"] + elapsed
                min_age = min(customer.get("min_age", age), age)
                max_age = max(customer.get("max_age", age), age)
                update_customer(name, {
                    "last_seen": now,
                    "visit_count": visits,
                    "total_time": total_time,
                    "min_age": min_age,
                    "max_age": max_age,
                    "gender" : gender
                })

            label = f"{name} | Age: {min_age}-{max_age} | Gender: {gender}"
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("SmartShop Tracker", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
