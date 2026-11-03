import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess(image_dir="data/raw/images", mask_dir="data/raw/masks", img_size=(256, 256)):
    """
    Loads raw satellite images and cloud masks, resizes them to 256x256,
    and normalizes pixel values to [0, 1].
    """
    images = []
    masks = []
    
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Directory Notice: Place images in '{image_dir}' and masks in '{mask_dir}'.")
        return np.array([]), np.array([])

    image_files = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])

    print(f"Found {len(image_files)} images and {len(mask_files)} masks.")

    for img_name, mask_name in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        # Load satellite image
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize RGB values
        
        # Load cloud mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, img_size)
        mask = mask / 255.0  # Normalize
        mask = np.expand_dims(mask, axis=-1)
        
        images.append(img)
        masks.append(mask)
        
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

def get_train_val_split(test_size=0.2):
    """Splits preprocessed data into training and validation sets."""
    X, y = load_and_preprocess()
    if len(X) == 0:
        print("No images found yet. Add image files to data/raw/ images and masks.")
        return None, None, None, None
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"Dataset Split Completed: {len(X_train)} Train, {len(X_val)} Validation")
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    print("Testing Data Preprocessing Pipeline...")
    get_train_val_split()