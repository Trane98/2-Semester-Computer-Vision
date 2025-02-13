import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r"2-Semester-Computer-Vision\Intro_for_Computer_vision.py\fruits_hsv.jpg", cv2.IMREAD_GRAYSCALE)

# Define the Horizontal Sobel Kernel
sobel_x = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]], dtype=np.float32)


# Define the Vertical Sobel Kernel
sobel_y = np.array([[+1, +2, 1], 
                    [0, 0, 0], 
                    [-1, -2, -1]], dtype=np.float32)


# Apply the kernel for horizontal edges
sobel_x_image = cv2.filter2D(image, -1, sobel_x)

# Apply the kernel for vertical edges
sobel_y_image = cv2.filter2D(image, -1, sobel_y)

# læg de to billeder sammen
sobel_xy_image = sobel_x_image + sobel_y_image


plt.figure(figsize=(15,10))

plt.subplot(1,4,1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(sobel_x_image, cmap="gray")
plt.title("Horizontal Sobel Edges")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(sobel_y_image, cmap="gray")
plt.title("Vertical Sobel Edges")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(sobel_xy_image, cmap="gray")
plt.title("Combined Sobel Edges")
plt.axis("off")

plt.show()



# ChatGBT løsning, hvor at der bruges sobel filters med OpenCV i stedet for filter2D for edge detection
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread(r"C:\Program Files (x86)\2 Semester python work\2-Semester-Computer-Vision\Intro_for_Computer_vision.py\fruits_hsv.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Sobel filters using OpenCV (better than filter2D for edge detection)
sobel_x_image = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobel_y_image = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

# Compute the gradient magnitude
sobel_xy_image = np.sqrt(sobel_x_image**2 + sobel_y_image**2)

# Normalize to range 0-255 for display
sobel_x_image = cv2.convertScaleAbs(sobel_x_image)  # Convert back to uint8
sobel_y_image = cv2.convertScaleAbs(sobel_y_image)
sobel_xy_image = cv2.convertScaleAbs(sobel_xy_image)

# Plot images
plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(sobel_x_image, cmap="gray")
plt.title("Horizontal Sobel Edges")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(sobel_y_image, cmap="gray")
plt.title("Vertical Sobel Edges")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(sobel_xy_image, cmap="gray")
plt.title("Combined Sobel Edges")
plt.axis("off")

plt.show()
