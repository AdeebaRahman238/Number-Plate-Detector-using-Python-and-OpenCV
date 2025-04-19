# main.py

import cv2
import pytesseract

# Step 1: Set the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # <- update if needed

# Step 2: Load the image
image_path = "car.jpg"  # <- make sure car.jpg is in the same folder as this script
image = cv2.imread(image_path)

# Step 3: Check if image was loaded successfully
if image is None:
    print("Error: Could not load image. Check if 'car.jpg' exists in the project folder.")
    exit()

# Step 4: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 5: Reduce noise and detect edges
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edges = cv2.Canny(gray, 170, 200)

# Step 6: Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

# Step 7: Find a rectangle contour (number plate)
number_plate_contour = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        number_plate_contour = approx
        break

# Step 8: If found, extract the plate region
if number_plate_contour is not None:
    cv2.drawContours(image, [number_plate_contour], -1, (0, 255, 0), 3)
    x, y, w, h = cv2.boundingRect(number_plate_contour)
    plate_image = image[y:y + h, x:x + w]

    # Optional: Show cropped plate
    cv2.imshow("Number Plate", plate_image)

    # Step 9: Extract text using OCR
    text = pytesseract.image_to_string(plate_image, config='--psm 8')
    print("Detected Number Plate Text:", text.strip())

else:
    print("Number plate contour not detected.")

# Step 10: Show final image
cv2.imshow("Detected Plate on Car", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
