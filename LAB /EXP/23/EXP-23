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

# Create blurred image
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Unsharp masking
sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

# Display
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(blurred, cmap='gray')
plt.title("Blurred Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sharpened, cmap='gray')
plt.title("Unsharp Masking")
plt.axis("off")

plt.show()
