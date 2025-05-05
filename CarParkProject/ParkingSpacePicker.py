import cv2
import pickle

width, height = 107, 48

# Load or initialize parking position list with compatibility for older format
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
        # Convert old format (list of positions) to new format (list of (position, name))
        if isinstance(posList[0], tuple) and isinstance(posList[0][0], int):  # Check if old format
            posList = [((x, y), f'p{i+1}') for i, (x, y) in enumerate(posList)]
except (FileNotFoundError, IndexError):
    posList = []


def mouseClick(events, x, y, flags, params):
    global posList
    # Add position on left-click
    if events == cv2.EVENT_LBUTTONDOWN:
        name = f'p{len(posList) + 1}'  # Generate unique name
        posList.append(((x, y), name))
    # Remove position on right-click
    elif events == cv2.EVENT_RBUTTONDOWN:
        for i, (pos, name) in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)
                break

    # Save updated positions to file
    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)


while True:
    # Try to load the image
    img = cv2.imread('carParkImg.png')
    if img is None:
        print("Error: Image file 'carParkImg.png' not found.")
        break

    # Draw all parking spots on the image with names
    for pos, name in posList:
        x, y = pos
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 255), 2)
        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display instructions on the image
    cv2.putText(img, "Left-click to add, Right-click to remove, 'q' to exit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the image and set mouse callback
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
