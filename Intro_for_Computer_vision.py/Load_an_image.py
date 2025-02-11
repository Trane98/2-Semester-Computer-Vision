import cv2

image = cv2.imread("fruits.jpg")
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = cv2.imread("fruits.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
