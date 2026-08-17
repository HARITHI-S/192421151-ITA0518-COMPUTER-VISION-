import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# INPUT IMAGE CREATED INSIDE CODE
# --------------------------------
img = np.zeros((300, 400), dtype=np.uint8)

# Create sample image
cv2.rectangle(img, (50, 50), (200, 200), 160, -1)
cv2.circle(img, (300, 150), 70, 220, -1)

# Add text
cv2.putText(img, "IMAGE", (90, 270),
            cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

# --------------------------------
# HIGH-BOOST MASK
# --------------------------------
A = 2

mask = np.array([
    [0, -1, 0],
    [-1, A + 4, -1],
    [0, -1, 0]
], dtype=np.float32)

# Apply high-boost mask
sharpened = cv2.filter2D(img, -1, mask)

# --------------------------------
# DISPLAY RESULTS
# --------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sharpened, cmap='gray')
plt.title("High-Boost Sharpened Image")
plt.axis("off")

plt.show()

# Display mask
print("High-Boost Mask:")
print(mask)
