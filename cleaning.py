import cv2
import os
import hashlib
from tqdm import tqdm

IMAGE_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"

def is_corrupted(img_path):
    img = cv2.imread(img_path)
    return img is None

def is_blurry(img, threshold=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def get_image_hash(img_path):
    with open(img_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

seen_hashes = set()

for img_name in tqdm(os.listdir(IMAGE_DIR)):
    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt"))

    # 1. Remove corrupted images
    if is_corrupted(img_path):
        print(f"Removing corrupted: {img_name}")
        os.remove(img_path)
        if os.path.exists(label_path):
            os.remove(label_path)
        continue

    img = cv2.imread(img_path)

    # 2. Remove blurry images
    if is_blurry(img):
        print(f"Removing blurry: {img_name}")
        os.remove(img_path)
        if os.path.exists(label_path):
            os.remove(label_path)
        continue

    # 3. Remove duplicate images
    img_hash = get_image_hash(img_path)
    if img_hash in seen_hashes:
        print(f"Removing duplicate: {img_name}")
        os.remove(img_path)
        if os.path.exists(label_path):
            os.remove(label_path)
        continue

    seen_hashes.add(img_hash)

print("Dataset cleaning completed!")
