# age_estimator.py

from deepface import DeepFace

def estimate_age(face_crop: any) -> int:
    """
    Uses DeepFace to estimate age. Handles list or dict return formats.
    """
    results = DeepFace.analyze(face_crop, actions=["age"], enforce_detection=False)
    # DeepFace may return a list of results
    if isinstance(results, list) and len(results) > 0:
        age_value = results[0].get("age", None)
    else:
        age_value = results.get("age", None)

    if age_value is None:
        return -1  # Could not estimate age
    return int(age_value)
