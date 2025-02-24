import cv2

# Open picture
python_logo = cv2.imread(r"C:\Program Files (x86)\2 Semester python work\2-Semester-Computer-Vision\Intro_for_Computer_vision.py\fruits.jpg", cv2.IMREAD_GRAYSCALE)

# Display the picture
cv2.imshow("Our window", python_logo)
cv2.waitKey(0)