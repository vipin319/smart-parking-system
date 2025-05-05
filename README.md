# Smart Parking Management

Welcome to the **Smart Parking Management System**! 🚗 This project leverages AI, computer vision, and real-time database management to streamline parking operations, ensuring a seamless and efficient experience for users.

---

## Features

- **Automated License Plate Detection**: 
  - Uses YOLO for object detection and EasyOCR for text recognition.
- **Real-Time Parking Slot Tracking**: 
  - Monitors and displays available slots dynamically.
- **Secure Database Management**: 
  - Stores vehicle entry and exit details with timestamps.
- **User-Friendly Interface**: 
  - Upload images and view records easily.
- **Detailed Parking Logs**: 
  - Tracks parking durations and provides accurate records.

---

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **AI Models**: YOLO (for object detection), EasyOCR (for text recognition)
- **Database**: SQLite
- **Other Tools**: OpenCV, Torch

---

## Prerequisites

Ensure you have the following installed:

- Python 3.7+
- pip (Python package manager)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/smart-parking-management.git
   ```
2. Navigate to the project directory:
   ```bash
   cd smart-parking-management
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Initialize the database:
   ```bash
   python app.py
   ```
   This will create an SQLite database file `parking_lot.db`.

---

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```
3. Use the interface to:
   - Upload vehicle entry and exit images.
   - View parking records.

---

## Directory Structure

```
smart-parking-management/
├── uploaded_images/         # Stores uploaded vehicle images
├── templates/               # HTML templates for the frontend
├── static/                  # Static files (CSS, JavaScript)
├── app.py                   # Flask application
├── requirements.txt         # Python dependencies
├── parking_lot.db           # SQLite database (auto-generated)
├── README.md                # Project documentation
```

---

## Acknowledgments

- YOLO and EasyOCR libraries for their powerful tools.
- Flask for its simplicity and efficiency in backend development.
- OpenCV for image processing capabilities.

---


Happy Parking! 🚗
