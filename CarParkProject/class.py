import cv2
import pickle
import cvzone
import numpy as np

# Load video feed
cap = cv2.VideoCapture('carPark.mp4')

# Load parking positions, classified into sections for different vehicle types
try:
    with open('CarParkPos', 'rb') as f:
        posDict = pickle.load(f)
except FileNotFoundError:
    posDict = {"car": [], "bus": [], "two_wheeler": []}  # Initialize with empty lists

width, height = 107, 48
threshold = 900  # Adjust this based on testing

def checkParkingSpace(imgPro, vehicle_type):
    spaceCounter = 0
    posList = posDict.get(vehicle_type, [])

    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        # Determine color and thickness based on parking status
        if count < threshold:
            color = (0, 255, 0)  # Green for free
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)  # Red for occupied
            thickness = 2

        # Draw rectangle and count text on each spot
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)

    # Display total free spaces for the selected vehicle type
    cvzone.putTextRect(img, f'{vehicle_type.capitalize()} Free: {spaceCounter}/{len(posList)}', 
                       (100, 50), scale=3, thickness=5, offset=20, colorR=(0, 200, 0))


# Prompt user to select vehicle type
vehicle_type = input("Enter the vehicle type (car, bus, two_wheeler): ").lower()
if vehicle_type not in posDict:
    print(f"Invalid vehicle type: {vehicle_type}. Defaulting to 'car'.")
    vehicle_type = "car"

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    if not success:
        break

    # Preprocess frame
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    # Check and display parking spaces based on vehicle type
    checkParkingSpace(imgDilate, vehicle_type)

    # Display the results
    cv2.imshow("Parking Detection", img)

    # Exit on pressing 'q'
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
