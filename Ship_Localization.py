import cv2
import imutils
import numpy as np

img1 = cv2.imread('Ships_Dataset/Scenes/lb_1.png')
img2 = cv2.imread('Ships_Dataset/Scenes/lb_4.png')
# img1 = cv2.imread('Ships_Dataset/Scenes/sfbay_2.png')
# img2 = cv2.imread('Ships_Dataset/Scenes/sfbay_3.png')

# Resize images if necessary
img1 = cv2.resize(img1, (600, 360))
img2 = cv2.resize(img2, (600, 360))

img_height = img1.shape[0]

# Grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

diff = cv2.absdiff(gray1, gray2)
cv2.imshow("Difference", diff)

thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Threshold", thresh)

# Dilation
kernel = np.ones((5, 5), np.uint8)
dilate = cv2.dilate(thresh, kernel, iterations=2)

# Calculate contours
contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

for contour in contours:
    if cv2.contourArea(contour) > 100:
        # Calculate bounding box around contour
        x, y, w, h = cv2.boundingRect(contour)
        # Draw rectangle - bounding box on both images
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Show images with rectangles on differences
x = np.zeros((img_height, 10, 3), np.uint8)
result = np.hstack((img1, x, img2))
cv2.imshow("Ship Localization Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
