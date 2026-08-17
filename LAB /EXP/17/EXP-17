import cv2
import numpy as np
import matplotlib.pyplot as plt
img = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 200), 255, -1)
cv2.circle(img, (300, 150), 80, 255, -1)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Input Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(sobel_x, cmap='gray')
plt.title("Sobel X-axis")
plt.axis("off")
plt.show()
