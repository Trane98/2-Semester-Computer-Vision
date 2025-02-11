import cv2
import numpy
from matplotlib import pyplot as plt
 

image = cv2.imread(r"C:\Program Files (x86)\2 Semester python work\2-Semester-Computer-Vision\Intro_for_Computer_vision.py\fruits_hsv.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Thresholding the image
ret, thresh_image = cv2.threshold(image, 131, 255, cv2.THRESH_BINARY)

# Show the thresholded image
cv2.imshow("Thresholded Image", thresh_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Finding Contours
contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for visualization
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)  # Draw contours in green

# Show the image with contours
cv2.imshow("Contours Detected", image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()