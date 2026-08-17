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

# Laplacian sharpening mask
mask = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

# Apply sharpening mask
sharpened = cv2.filter2D(img, -1, mask)

# Display
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sharpened, cmap='gray')
plt.title("Sharpened Image")
plt.axis("off")

plt.show()
