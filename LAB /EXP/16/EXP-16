import cv2
import numpy as np
import matplotlib.pyplot as plt
img = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 200), 255, -1)
cv2.circle(img, (300, 150), 80, 255, -1)
edges = cv2.Canny(img, 100, 200)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Input Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis("off")
plt.show()
