import cv2
import os
import random
import numpy as np

INPUT_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"
OUTPUT_IMG_DIR = "augmented/images"
OUTPUT_LABEL_DIR = "augmented/labels"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def load_labels(label_path):
    if not os.path.exists(label_path):
        return []
    with open(label_path, "r") as f:
        return [list(map(float, line.strip().split())) for line in f.readlines()]

def save_labels(label_path, labels):
    with open(label_path, "w") as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")

# Rotate image and labels
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

def adjust_exposure(image):
    factor = random.uniform(0.94, 1.06)  # ±6%
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def apply_blur(image):
    k = random.choice([1, 3, 5])
    return cv2.GaussianBlur(image, (k, k), 0)

for img_name in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt"))

    image = cv2.imread(img_path)
    if image is None:
        continue

    labels = load_labels(label_path)

    base_name = img_name.split(".")[0]

    # Save original
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, img_name), image)
    save_labels(os.path.join(OUTPUT_LABEL_DIR, base_name + ".txt"), labels)

    # 1. 90° rotations
    for angle in [90, -90, 180]:
        aug_img = rotate_image(image, angle)
        new_name = f"{base_name}_rot{angle}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, new_name), aug_img)
        save_labels(os.path.join(OUTPUT_LABEL_DIR, new_name.replace(".jpg", ".txt")), labels)

    # 2. Random rotation (-15 to 15)
    angle = random.uniform(-15, 15)
    aug_img = rotate_image(image, angle)
    new_name = f"{base_name}_randrot.jpg"
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, new_name), aug_img)
    save_labels(os.path.join(OUTPUT_LABEL_DIR, new_name.replace(".jpg", ".txt")), labels)

    # 3. Exposure adjustment
    aug_img = adjust_exposure(image)
    new_name = f"{base_name}_exposure.jpg"
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, new_name), aug_img)
    save_labels(os.path.join(OUTPUT_LABEL_DIR, new_name.replace(".jpg", ".txt")), labels)

    # 4. Gaussian blur
    aug_img = apply_blur(image)
    new_name = f"{base_name}_blur.jpg"
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, new_name), aug_img)
    save_labels(os.path.join(OUTPUT_LABEL_DIR, new_name.replace(".jpg", ".txt")), labels)

print("Augmentation completed!")
