# utils.py

import os
import cv2
import datetime

IMAGE_DIR = "images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


def save_image(crop: any, name: str) -> str:
    """
    Saves a cropped image under images/<name>.jpg
    """
    filename = os.path.join(IMAGE_DIR, f"{name}.jpg")
    cv2.imwrite(filename, crop)
    return filename

def get_timestamp() -> datetime.datetime:
    """
    Returns the current datetime for visit logging.
    """
    return datetime.datetime.now()
