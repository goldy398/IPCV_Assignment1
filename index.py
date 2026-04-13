"""
Name: Lakshay
Roll No: 2301010306
Course: Image Processing & Computer Vision
Unit: Mini Project
Assignment Title: Smart Document Scanner & Quality Analysis System
Date: 
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("📄 Welcome to Smart Document Scanner & Quality Analysis System")

# Create output folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# -----------------------------
# Task 2: Image Acquisition
# -----------------------------
# Load image (replace with your file path)
image_path = "sample.jpg"
img = cv2.imread(image_path)

if img is None:
    print("❌ Error: Image not found")
    exit()

# Resize to 512x512
img_resized = cv2.resize(img, (512, 512))

# Convert to grayscale
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Task 3: Sampling (Resolution)
# -----------------------------
def resize_and_upscale(image, size):
    down = cv2.resize(image, size)
    up = cv2.resize(down, (512, 512))
    return up

high_res = gray  # 512x512
med_res = resize_and_upscale(gray, (256, 256))
low_res = resize_and_upscale(gray, (128, 128))

# -----------------------------
# Task 4: Quantization
# -----------------------------
def quantize(image, levels):
    factor = 256 // levels
    quantized = (image // factor) * factor
    return quantized

q_8bit = gray  # 256 levels
q_4bit = quantize(gray, 16)
q_2bit = quantize(gray, 4)

# -----------------------------
# Save outputs
# -----------------------------
cv2.imwrite("outputs/original.png", img_resized)
cv2.imwrite("outputs/grayscale.png", gray)
cv2.imwrite("outputs/high_res.png", high_res)
cv2.imwrite("outputs/med_res.png", med_res)
cv2.imwrite("outputs/low_res.png", low_res)
cv2.imwrite("outputs/q_8bit.png", q_8bit)
cv2.imwrite("outputs/q_4bit.png", q_4bit)
cv2.imwrite("outputs/q_2bit.png", q_2bit)

# -----------------------------
# Task 5: Visualization
# -----------------------------
titles = [
    "Original", "Grayscale",
    "512x512", "256x256", "128x128",
    "8-bit", "4-bit", "2-bit"
]

images = [
    cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), gray,
    high_res, med_res, low_res,
    q_8bit, q_4bit, q_2bit
]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig("outputs/comparison.png")
plt.show()

# -----------------------------
# Observations
# -----------------------------
print("\n📊 Observations:")
print("1. High resolution (512x512) retains fine details.")
print("2. Medium resolution shows slight blurring.")
print("3. Low resolution loses text clarity significantly.")
print("4. 8-bit quantization preserves quality.")
print("5. 4-bit introduces visible banding.")
print("6. 2-bit severely degrades readability.")
print("7. OCR works best with high resolution and higher bit-depth.")
