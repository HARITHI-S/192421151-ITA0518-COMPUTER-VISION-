import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input image created inside the code
img = np.zeros((300, 400), dtype=np.uint8)

# Create sample image
cv2.rectangle(img, (50, 50), (200, 200), 180, -1)
cv2.circle(img, (300, 150), 70, 220, -1)

# Add text
cv2.putText(img, "IMAGE", (90, 270),
            cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

# Laplacian mask with diagonal neighbors
mask = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
], dtype=np.float32)

# Apply mask
laplacian = cv2.filter2D(img, cv2.CV_64F, mask)

# Sharpen image
sharpened = img.astype(np.float64) - laplacian

# Convert to 8-bit
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

# Display
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(np.abs(laplacian), cmap='gray')
plt.title("Laplacian")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sharpened, cmap='gray')
plt.title("Sharpened Image")
plt.axis("off")

plt.show()
