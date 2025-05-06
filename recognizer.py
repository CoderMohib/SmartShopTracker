#recognizer.py
import numpy as np
from deepface import DeepFace
from typing import Tuple, Optional

# Global stores for embeddings & names
known_embeddings = []
known_names = []

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(crop: any, thresh: float = 0.4) -> Tuple[Optional[str], Optional[np.ndarray], bool]:
    """
    Attempts to detect & embed a face in `crop` using DeepFace with enforce_detection=True.
    Returns:
      - name: matched person name, or None if unrecognized
      - embedding: face embedding, or None if no face
      - face_found: True only if a real face was detected & embedded
    """
    try:
        # Strict detection: will raise if no face is confidently found
        reps = DeepFace.represent(crop, model_name="Facenet", enforce_detection=True)
    except Exception:
        return None, None, False

    if not reps or "embedding" not in reps[0]:
        return None, None, False

    emb = np.asarray(reps[0]["embedding"], dtype=np.float32)

    # Find best existing match
    best_match = None
    best_dist = thresh
    for name, known_emb in zip(known_names, known_embeddings):
        dist = cosine_distance(emb, known_emb)
        if dist < best_dist:
            best_dist = dist
            best_match = name

    return best_match, emb, True
