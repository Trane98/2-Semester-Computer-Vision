import cv2
import numpy as np
import matplotlib.pyplot as plt

# Indlæs billedet
image = cv2.imread(r"C:\Program Files (x86)\2 Semester python work\2-Semester-Computer-Vision\Intro_for_Computer_vision.py\fruits.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konverter til RGB
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Konverter til HSV

# Beregn gennemsnitlig farve (RGB)
mean_color_rgb = np.mean(image_rgb, axis=(0,1))
mean_color_hsv = np.mean(image_hsv, axis=(0,1))

print(f"Gennemsnitlig RGB-farve: {mean_color_rgb}")
print(f"Gennemsnitlig HSV-farve: {mean_color_hsv}")

# Plot originalbillede
plt.figure(figsize=(5,5))
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Original Image")
plt.show()


# Konverter til gråskala
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find kanter med Canny
edges = cv2.Canny(gray, 50, 150)

# Find konturer
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tegn konturer på billedet
image_contours = image_rgb.copy()
cv2.drawContours(image_contours, contours, -1, (0,255,0), 2)

# Plot konturbilledet
plt.figure(figsize=(5,5))
plt.imshow(image_contours)
plt.axis("off")
plt.title("Detected Contours")
plt.show()



# Gennemgå hver kontur og beregn features
form_features = []
for contour in contours:
    # Beregn areal
    area = cv2.contourArea(contour)
    
    # Beregn bounding box og aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    
    # Beregn cirkularitet (1 = perfekt cirkel, <1 = ikke cirkulær)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

    form_features.append((area, aspect_ratio, circularity))

# Print formfeatures
for i, (area, aspect_ratio, circularity) in enumerate(form_features):
    print(f"Objekt {i+1}: Areal={area:.2f}, Aspect Ratio={aspect_ratio:.2f}, Cirkularitet={circularity:.2f}")



from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Opret feature-matrix (farve + form)
X = np.array(form_features)  # Vi kan også tilføje mean_color_rgb her
y = np.array([0, 0, 1])  # Antag æbler=0, banan=1

# Split data i træning/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Normaliser features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Træn en SVM-model
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)

# Test modellen
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")
