from deepface import DeepFace

def estimate_age(face_crop: any) -> int:
    results = DeepFace.analyze(face_crop, actions=["age"], enforce_detection=False)
    if isinstance(results, list) and len(results) > 0:
        age_value = results[0].get("age", None)
    else:
        age_value = results.get("age", None)
    if age_value is None:
        return -1
    return int(age_value)
