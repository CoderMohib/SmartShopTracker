from deepface import DeepFace
import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")

    # Resize to 224x224
    img = cv2.resize(img, (224, 224))

    # Enhance contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    img_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    return img_eq

def analyze_gender(image_path):
    try:
        # Preprocess image
        img = preprocess_image(image_path)

        # Run DeepFace analysis
        result = DeepFace.analyze(
            img,
            actions=["gender"],
            enforce_detection=False
        )

        if isinstance(result, list):
            result = result[0]

        # Handle both dict or string type result
        gender_data = result.get("gender", {})
        if isinstance(gender_data, dict):
            return gender_data
        else:
            # If it's a string like "Man" or "Woman"
            return {"Man": 100.0, "Woman": 0.0} if gender_data == "Man" else {"Man": 0.0, "Woman": 100.0}

    except Exception as e:
        print(f"Error: {e}")
        return {}

# === USAGE EXAMPLE ===
if __name__ == "__main__":
    image_path = "images/tete.jpg"
    gender_scores = analyze_gender(image_path)
    print("Gender confidence scores:", gender_scores)
