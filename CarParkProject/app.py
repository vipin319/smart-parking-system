from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import cvzone
import numpy as np

app = Flask(__name__)

# Video feed
cap = cv2.VideoCapture('carPark.mp4')

# Load parking positions with names
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
        posList = [(pos[0], pos[1]) for pos in posList]
except Exception as e:
    print(f"Error loading positions: {e}")
    posList = []

if not posList:
    print("Error: No valid positions loaded. Exiting.")
    exit()

width, height = 107, 48
threshold = 900

def checkParkingSpace(img, imgPro):
    spaceCounter = 0
    empty_slots = []  # List to store names of empty slots
    for pos, name in posList:
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < threshold:
            color = (0, 255, 0)  # Green for free
            thickness = 5
            spaceCounter += 1
            empty_slots.append(name)  # Add name to empty slots
        else:
            color = (0, 0, 255)  # Red for occupied
            thickness = 2

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness)
        # Draw parking space name
        cvzone.putTextRect(img, name, (x, y - 5), scale=1,
                           thickness=2, offset=5, colorR=color)

    return spaceCounter, empty_slots  # Return both count and empty slots

def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            # Reset to the beginning of the video for looping
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, img = cap.read()
            if not success:  # If reading still fails, break the loop
                break

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        spaceCounter, empty_slots = checkParkingSpace(img, imgDilate)

        # Display total free spaces
        cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                           thickness=5, offset=20, colorR=(0, 200, 0))

        # Encode the image as JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/empty_slots')
def empty_slots():
    success, img = cap.read()
    if success:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        spaceCounter, empty_slots_list = checkParkingSpace(img, imgDilate)
        return jsonify(empty_slots=empty_slots_list)
    return jsonify(empty_slots=[])

if __name__ == '__main__':
    app.run(debug=True,port=4000)