# importing the modules needed
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the image
image = cv2.imread(r"C:\Program Files (x86)\2 Semester python work\2-Semester-Computer-Vision\Intro_for_Computer_vision.py\fruits.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

kernel_size = 10  # You can change this to 5, 7, etc., for stronger blurring
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)


blurred_image = cv2.filter2D(image, -1, kernel)


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(blurred_image)
plt.title("Blurred Image (Mean Kernel)")
plt.axis("off")

plt.show()