import cv2
import numpy as np

input_image = cv2.imread(r"C:\Program Files (x86)\2 Semester python work\2-Semester-Computer-Vision\Intro_for_Computer_vision.py\fruits.jpg", cv2.IMREAD_GRAYSCALE)
output_image = np.zeros(input_image.shape, dtype=input_image.dtype)

threshold = 150

for y, row in enumerate(input_image):
    for x, pixel in enumerate(row):
        if pixel > threshold:
            output_image[y, x] = 255

mask_output = (input_image > threshold) * np.uint8(255)
print(mask_output.dtype)

overwrite_output = input_image.copy()
overwrite_output[overwrite_output > threshold] = 255
overwrite_output[overwrite_output < 255] = 0

cv2.imshow("Input image", input_image)
cv2.imshow("Output image", output_image)
cv2.imshow("Mask output", mask_output)
cv2.imshow("Overwrite output", overwrite_output)
cv2.waitKey(0)