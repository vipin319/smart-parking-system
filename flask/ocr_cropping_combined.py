import os
import cv2
import torch
from ultralytics import YOLO
import easyocr
import warnings
import sys
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress YOLO logs
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)

argument = ""
if len(sys.argv) > 1:
    argument = sys.argv[1]

# Load the YOLO model
model = YOLO('ultralytics/runs/detect/train_model/weights/best.pt')

# Perform prediction
results = model.predict(f'./images/{argument}', save=False, imgsz=320, conf=0.2, verbose=False)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Character correction dictionary
correction_dict = {
    'Z': '2',
    '[': 'T',
    'O': '0',
    'S': '5',
    'I': '1',
    'B': '8'
}

def preprocess_image(cropped_img):
    """
    Preprocess the cropped image for better OCR results.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    processed = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return processed

def correct_text(text):
    """
    Correct OCR text using a character correction dictionary.
    """
    return ''.join(correction_dict.get(char, char) for char in text)

# Extract the detected bounding boxes
for result in results:
    boxes = result.cpu().numpy() if isinstance(result, torch.Tensor) else result.boxes
    img = cv2.imread(f'./images/{argument}')

    for i in range(boxes.shape[0]):
        box = boxes[i]
        x1, y1, x2, y2, conf = map(int, box[:5])
        cropped_img = img[y1:y2, x1:x2]
        
        # Preprocess the cropped image
        processed_img = preprocess_image(cropped_img)
        
        # Perform OCR on the preprocessed image
        ocr_results = reader.readtext(processed_img)
        
        # Merge the text and correct it
        merged_text = ''.join(text for (_, text, _) in ocr_results).replace(" ", "").upper()
        corrected_text = correct_text(merged_text)
        
        # Print only the corrected text
        print(corrected_text)
