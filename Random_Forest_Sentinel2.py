import os
from glob import glob
import numpy as np
import rasterio
from tqdm import tqdm

tiles_root = r"E:\Aishwarya Gilhotra\Project\Dataset\TILES"
masks_root = r"E:\Aishwarya Gilhotra\Project\Dataset\MASKS"

def get_suffix(filename):
    parts = filename.replace(".tif", "").split("_")
    return parts[-2] + "_" + parts[-1]

pairs = []

tile_folders = os.listdir(tiles_root)

for folder in tile_folders:
    tile_folder_path = os.path.join(tiles_root, folder)
    mask_folder_path = os.path.join(masks_root, folder)

    if not os.path.exists(mask_folder_path):
        print(f"Missing mask folder for: {folder}")
        continue

    tile_files = glob(os.path.join(tile_folder_path, "*.tif"))
    mask_files = glob(os.path.join(mask_folder_path, "*.tif"))

    # Build mask lookup
    mask_dict = {}
    for m in mask_files:
        suffix = get_suffix(os.path.basename(m))
        mask_dict[suffix] = m

    # Match tiles
    for t in tile_files:
        suffix = get_suffix(os.path.basename(t))
        if suffix in mask_dict:
            pairs.append((t, mask_dict[suffix]))
        else:
            print(f"⚠️ No mask for: {t}")

print("Total pairs:", len(pairs))

X_list = []
y_list = []

for tile_path, mask_path in tqdm(pairs):

    with rasterio.open(tile_path) as src:
        tile = src.read()

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    tile_reshaped = tile.reshape(tile.shape[0], -1).T
    mask_reshaped = mask.flatten()

    valid = mask_reshaped >= 0

    X_valid = tile_reshaped[valid]
    y_valid = mask_reshaped[valid]

    # 🔥 SAMPLE ONLY 2% (start small)
    n_samples = int(0.02 * len(y_valid))

    if n_samples == 0:
        continue

    indices = np.random.choice(len(y_valid), n_samples, replace=False)

    X_list.append(X_valid[indices])
    y_list.append(y_valid[indices])

X = np.vstack(X_list)
y = np.hstack(y_list)

# Convert labels to int
y = y.astype(np.int32)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Unique labels:", np.unique(y))
print("NaN values in X:", np.isnan(X).sum())

print("Class distribution:", np.bincount(y))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Report
print(classification_report(y_test, y_pred))

import pandas as pd

features = ["B2", "B3", "B4", "B8", "NDVI", "NDWI", "SAVI"]

importances = model.feature_importances_

for f, imp in zip(features, importances):
    print(f"{f}: {imp:.4f}")

import matplotlib.pyplot as plt

# Pick one sample tile
tile_path, mask_path = pairs[0]

# Read tile
with rasterio.open(tile_path) as src:
    tile = src.read()  # (7, H, W)

# Read mask
with rasterio.open(mask_path) as src:
    mask = src.read(1)

# Create RGB image
R = tile[2]
G = tile[1]
B = tile[0]

rgb = np.stack([R, G, B], axis=-1)

# Normalize for display
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

# Prepare data for prediction
H, W = mask.shape

tile_reshaped = tile.reshape(tile.shape[0], -1).T

# Predict
pred = model.predict(tile_reshaped)

# Reshape back
pred_mask = pred.reshape(H, W)

# Plot
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("RGB Image")
plt.imshow(rgb)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Ground Truth Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Model Prediction")
plt.imshow(pred_mask, cmap="gray")
plt.axis("off")

plt.show()
