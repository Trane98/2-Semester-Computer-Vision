import cv2
import itertools

image = cv2.imread(r"C:\Program Files (x86)\2 Semester python work\2-Semester-Computer-Vision\Intro_for_Computer_vision.py\fruits.jpg", cv2.IMREAD_GRAYSCALE)

for row in image:
    for pixel in row:
        print(f"Pixel value: {pixel}")

for y, row in enumerate(image):
    for x, pixel in enumerate(row):
        print(f"Pixel value at ({x}, {y}): {pixel}")

for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        print(f"Pixel value with range at ({x}, {y}): {image[y, x]}")

for y, x in itertools.product(range(image.shape[0]), range(image.shape[1])):
    print(f"Pixel value with itertools at ({x}, {y}): {image[y, x]}")