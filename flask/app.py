import os
import cv2
import torch
from ultralytics import YOLO
import easyocr
import warnings
from flask import Flask, request, render_template, jsonify
import sqlite3
import logging
from datetime import datetime

# Suppress warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model and EasyOCR reader
model = YOLO('ultralytics/runs/detect/train_model/weights/best.pt')
reader = easyocr.Reader(['en'], gpu=True)

# Database initialization
DB_FILE = 'parking_lot.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            noplate TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT DEFAULT NULL
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-entry', methods=['POST'])
def upload_entry():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Perform prediction
        results = model.predict(file_path, save=False, imgsz=320, conf=0.2, verbose=False)
        detected_texts = []

        img = cv2.imread(file_path)
        for result in results:
            boxes = result.cpu().numpy() if isinstance(result, torch.Tensor) else result.boxes
            for i in range(boxes.shape[0]):
                box = boxes[i]
                x1, y1, x2, y2, conf = map(int, box[:5])
                cropped_img = img[y1:y2, x1:x2]
                ocr_results = reader.readtext(cropped_img)
                merged_text = ''.join(text for (_, text, _) in ocr_results).replace(" ", "").upper()
                detected_texts.append(merged_text)

        # Insert into database
        if detected_texts:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            for text in detected_texts:
                cursor.execute('INSERT INTO vehicles (noplate, entry_time) VALUES (?, ?)', 
                               (text, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            conn.close()

        return jsonify({'message': 'Vehicle entered, data updated to database', 'detected_texts': detected_texts})
@app.route('/get-records', methods=['GET'])
def get_records():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT id, noplate, entry_time, exit_time FROM vehicles')
        rows = cursor.fetchall()
        conn.close()

        records = [
            {
                'id': row[0],
                'noplate': row[1],
                'entry_time': row[2],
                'exit_time': row[3]
            }
            for row in rows
        ]
        return jsonify({'records': records})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/upload-exit', methods=['POST'])
def upload_exit():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Perform OCR and prediction
        results = model.predict(file_path, save=False, imgsz=320, conf=0.2, verbose=False)
        detected_texts = []

        img = cv2.imread(file_path)
        for result in results:
            boxes = result.cpu().numpy() if isinstance(result, torch.Tensor) else result.boxes
            for i in range(boxes.shape[0]):
                box = boxes[i]
                x1, y1, x2, y2, conf = map(int, box[:5])
                cropped_img = img[y1:y2, x1:x2]
                ocr_results = reader.readtext(cropped_img)
                merged_text = ''.join(text for (_, text, _) in ocr_results).replace(" ", "").upper()
                detected_texts.append(merged_text)

        # Check database for matches
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        matched_text = None
        duration = None

        if detected_texts:
            for text in detected_texts:
                cursor.execute('SELECT id, entry_time FROM vehicles WHERE noplate = ? AND exit_time IS NULL', (text,))
                row = cursor.fetchone()
                if row:
                    matched_text = text
                    entry_time = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
                    exit_time = datetime.now()
                    duration = exit_time - entry_time

                    # Update exit time
                    cursor.execute('UPDATE vehicles SET exit_time = ? WHERE id = ?', 
                                   (exit_time.strftime('%Y-%m-%d %H:%M:%S'), row[0]))
                    conn.commit()
                    break

        conn.close()

        if matched_text:
            return jsonify({'message': f'Vehicle {matched_text} exited, total time: {duration}', 'duration': str(duration)})
        else:
            return jsonify({'error': 'No matching vehicle found in the database'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
