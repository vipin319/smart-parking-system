import cv2
import pickle
import cvzone
import numpy as np

# Video feed
cap = cv2.VideoCapture('carPark.mp4')

# Load parking positions with names
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
        print(f"Loaded raw positions: {posList}")  # Debugging
        # Extract positions and names separately
        posList = [(pos[0], pos[1]) for pos in posList]
except Exception as e:
    print(f"Error loading positions: {e}")
    posList = []

if not posList:
    print("Error: No valid positions loaded. Exiting.")
    exit()

width, height = 107, 48
threshold = 900

def checkParkingSpace(imgPro):
    spaceCounter = 0

    for pos, name in posList:
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < threshold:
            color = (0, 255, 0)  # Green for free
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)  # Red for occupied
            thickness = 2

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness)
        # Draw parking space name
        cvzone.putTextRect(img, name, (x, y - 5), scale=1,
                           thickness=2, offset=5, colorR=color)

    # Display total free spaces
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0, 200, 0))

while True:
    success, img = cap.read()
    if not success:
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    checkParkingSpace(imgDilate)

    cv2.imshow("Parking Detection", img)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
