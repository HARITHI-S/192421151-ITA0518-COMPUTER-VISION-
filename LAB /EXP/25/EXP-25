import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# INPUT IMAGE CREATED INSIDE CODE
# --------------------------------
img = np.zeros((300, 400), dtype=np.uint8)

# Create sample objects
cv2.rectangle(img, (50, 50), (200, 200), 160, -1)
cv2.circle(img, (300, 150), 70, 220, -1)

# Add text
cv2.putText(img, "IMAGE", (90, 270),
            cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

# --------------------------------
# GRADIENT MASKS
# --------------------------------

# X-direction mask
mask_x = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
], dtype=np.float32)

# Y-direction mask
mask_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# --------------------------------
# APPLY GRADIENT MASKS
# --------------------------------

gradient_x = cv2.filter2D(img, cv2.CV_64F, mask_x)
gradient_y = cv2.filter2D(img, cv2.CV_64F, mask_y)

# Calculate gradient magnitude
gradient = np.sqrt(gradient_x**2 + gradient_y**2)

# Convert gradient to 8-bit
gradient = cv2.convertScaleAbs(gradient)

# --------------------------------
# SHARPENING
# --------------------------------

sharpened = cv2.add(img, gradient)

# --------------------------------
# DISPLAY RESULTS
# --------------------------------

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gradient, cmap='gray')
plt.title("Gradient Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sharpened, cmap='gray')
plt.title("Sharpened Image")
plt.axis("off")

plt.show()

# Print masks
print("X-direction Gradient Mask:")
print(mask_x)

print("\nY-direction Gradient Mask:")
print(mask_y)
