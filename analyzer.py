from deepface import DeepFace
from database import get_customer_by_name, update_customer

def analyze_face(face_crop: any, customer_name: str) -> dict:
    """
    Uses DeepFace to estimate age and gender with bias correction.
    Caches analyzed data in the database.
    """
    try:
        # Check if customer data exists
        customer = get_customer_by_name(customer_name)
        if customer:
            min_age = customer.get("min_age", -1)
            max_age = customer.get("max_age", -1)
            gender_result = customer.get("gender", {})

            # Determine gender from dict or string
            if isinstance(gender_result, str):
                gender = gender_result
            elif isinstance(gender_result, dict) and gender_result:
                man_score = gender_result.get("Man", 0)
                woman_score = gender_result.get("Woman", 0)
                gender = "Man" if man_score >= woman_score else "Woman"
            else:
                gender = "Unknown"

            return {
                "min_age": min_age,
                "max_age": max_age,
                "gender": gender
            }

        # Run analysis if not found in database
        result = DeepFace.analyze(face_crop, actions=["age", "gender"], models={"gender": "FairFace", "age": "FairFace"}, enforce_detection=False)
        if isinstance(result, list):
            result = result[0]

        age = int(result.get("age", -1))
        gender_result = result.get("gender", {})

        # Bias correction
        if age < 40:
            age = max(0, age - 2)

        # Compute age range
        min_age = max(0, age - 3)
        max_age = age + 3

        # Determine gender
        if isinstance(gender_result, dict) and gender_result:
            man_score = gender_result.get("Man", 0)
            woman_score = gender_result.get("Woman", 0)
            gender = "Man" if man_score >= woman_score else "Woman"
        elif isinstance(gender_result, str):
            gender = gender_result
            gender_result = {"Man": 100.0, "Woman": 0.0} if gender == "Man" else {"Man": 0.0, "Woman": 100.0}
        else:
            gender = "Unknown"
            gender_result = {}

        # Save to DB
        data = {
            "name": customer_name,
            "min_age": min_age,
            "max_age": max_age,
            "gender": gender_result
        }
        update_customer(customer_name, data)

        return {
            "min_age": min_age,
            "max_age": max_age,
            "gender": gender
        }

    except Exception as e:
        print(f"Error during analysis: {e}")
        return {
            "min_age": -1,
            "max_age": -1,
            "gender": "Unknown"
        }
