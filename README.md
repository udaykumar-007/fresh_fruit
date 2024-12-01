# Fruit & Vegetable Freshness Detection App

This application detects the freshness of fruits and vegetables using AI-powered object detection and analysis. It supports both Python-based systems and Android devices for seamless image capturing, object detection, and freshness evaluation.

---

## Features

- **AI Object Detection:**  
  Detects fruits and vegetables using the YOLOv8s model.  
  - **Python Version:** YOLOv8s for detailed and accurate analysis.  
  - **Android Version:** YOLOv8-nano (ONNX) for lightweight and faster detection on mobile devices.  

- **Freshness Analysis:**  
  Assesses freshness using the Google Gemini API integrated via LangChain.  
  - **Result Indicators:**  
    - 🟢 **Green:** Fresh  
    - 🔴 **Red:** Not Fresh  

- **Cross-Platform Support:**  
  - **Python App:** Built with Kivy for image capture and displaying results.  
  - **Android App:** Optimized for mobile with threaded Gemini API requests.

---

## Requirements

### Python
- Python 3.x  
- Kivy  
- YOLOv8s model (ONNX compatible for Android)  
- Gemini API integration (via LangChain)  
- OpenCV or similar image processing library  

### Android
- Android Studio  
- YOLOv8-nano ONNX model for object detection  
- Gemini API for freshness analysis  

---

## Installation(Python)

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/fruit-vegetable-freshness-detection.git
    cd fruit-vegetable-freshness-detection
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. If you haven't already, download the YOLOv8s model.

## Usage

1. Run the app:
    ```bash
    python app.py
    ```

2. Capture an image of a fruit or vegetable using the Kivy interface.

3. The app will detect the objects in the image and then use the Gemini API to analyze if the detected items are fresh.

4. The freshness status will be displayed on the screen.

## Installation(Android)
Download and install apk using the link given in the repository. 

## Demo Video

Demo video is given in repo.


## Acknowledgements
- YOLOv8s for object detection
- Kivy for the mobile interface
- Google Gemini for freshness classification

## Image

![App Screenshot](python-kivy/test1.jpg)
