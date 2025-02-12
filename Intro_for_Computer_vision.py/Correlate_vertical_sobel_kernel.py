import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r"C:\Program Files (x86)\2 Semester python work\2-Semester-Computer-Vision\Intro_for_Computer_vision.py\fruits_hsv.jpg", cv2.IMREAD_GRAYSCALE)

# Define the Vertical Sobel Kernel
sobel_x = np.array([[+1, +2, 1], 
                    [0, 0, 0], 
                    [-1, -2, -1]], dtype=np.float32)

# Apply the kernel
sobel_x_image = cv2.filter2D(image, -1, sobel_x)



plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sobel_x_image, cmap="gray")
plt.title("Vertical Sobel Edges")
plt.axis("off")

plt.show()
