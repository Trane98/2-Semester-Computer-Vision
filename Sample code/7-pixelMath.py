import cv2
import numpy as np

input_image = cv2.imread(r"C:\Program Files (x86)\2 Semester python work\2-Semester-Computer-Vision\Intro_for_Computer_vision.py\fruits.jpg")
output_image = np.zeros(input_image.shape, dtype=input_image.dtype)

for y, row in enumerate(input_image):
    for x, pixel in enumerate(row):
        output_image[y, x] = pixel + 30

matrix_output = input_image + 30

cv2.imshow("Input image", input_image)
cv2.imshow("Output image", output_image)
cv2.imshow("Matrix output", matrix_output)
cv2.waitKey(0)