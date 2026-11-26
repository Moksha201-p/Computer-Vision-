import cv2
import numpy as np
from pathlib import Path
from modules.enhancement import ZeroDCEEnhancer


# ==========================================
# PROJECT DIRECTORY CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "dataset" / "low_light"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# FIND INPUT IMAGES
# ==========================================

image_files = [
    file for file in INPUT_DIR.iterdir()
    if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

if not image_files:
    raise FileNotFoundError(
        "No images found in dataset/low_light folder."
    )

image_path = image_files[0]

print("\nSelected image:", image_path.name)


# ==========================================
# LOAD IMAGE (UNICODE-SAFE)
# ==========================================

image_data = np.fromfile(
    str(image_path),
    dtype=np.uint8
)

original = cv2.imdecode(
    image_data,
    cv2.IMREAD_COLOR
)

if original is None:
    raise ValueError("Unable to read the input image.")

print("Original image loaded successfully!")
print("Image dimensions:", original.shape)


# ==========================================
# LOAD PRETRAINED AI MODEL
# ==========================================

print("\nLoading Zero-DCE++ AI model...")

enhancer = ZeroDCEEnhancer()

print("AI model loaded successfully!")


# ==========================================
# PERFORM AI IMAGE ENHANCEMENT
# ==========================================

print("\nApplying AI enhancement...")

enhanced = enhancer.enhance(original)

print("Image enhancement completed!")


# ==========================================
# SAVE ENHANCED IMAGE (UNICODE-SAFE)
# ==========================================

output_path = OUTPUT_DIR / f"enhanced_{image_path.name}"

success, encoded_image = cv2.imencode(
    output_path.suffix,
    enhanced
)

if not success:
    raise RuntimeError("Unable to encode enhanced image.")

encoded_image.tofile(str(output_path))


# ==========================================
# FINAL RESULTS
# ==========================================

print("\n======================================")
print("       AI ENHANCEMENT COMPLETED")
print("======================================")

print("Original Image :", image_path.name)
print("Enhanced Image :", output_path.name)
print("Image Size     :", enhanced.shape)
print("Processing     : Zero-DCE++ AI")
print("Device         : CPU")
print("Output Folder  :", output_path)

print("======================================")
