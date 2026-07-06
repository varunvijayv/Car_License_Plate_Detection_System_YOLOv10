# Car License Plate Detection System

An Automatic License Plate Recognition (ALPR) system that combines **YOLOv10** for license plate detection and **PaddleOCR** for text extraction, with results logged to both **SQLite** and **JSON** for record management.

## Overview

Manual identification of vehicle license plates from images or surveillance footage is slow and error-prone, especially in large-scale monitoring environments such as traffic management, toll booths, and parking systems. Traditional image-processing approaches often struggle with varying lighting, motion blur, and diverse plate formats.

This project addresses that gap with a deep learning pipeline that:
- Detects and localizes license plates in images and video frames using a custom-trained **YOLOv10-nano** model.
- Extracts alphanumeric text from the detected plate regions using **PaddleOCR**.
- Cleans and validates the recognized text.
- Stores results (plate number + timestamps) in a **SQLite** database and **JSON** log files.
- Overlays bounding boxes and recognized text on the image/video using **OpenCV** for real-time visualization.

## Features

- Supports both **image mode** and **video mode** as input.
- Real-time detection and recognition suitable for continuous surveillance.
- Handles varied lighting conditions, motion blur, partial occlusion, and oblique plate angles reasonably well.
- Dual persistence layer: SQLite database (structured records with timestamps) + JSON files (cumulative logs).
- Visual verification via bounding box and text overlay on frames.

## System Workflow

1. **Dataset Preparation** – Annotated license plate dataset sourced from Roboflow, containing diverse samples across lighting and environmental conditions.
2. **Model Selection & Training** – YOLOv10-nano fine-tuned via transfer learning from pretrained YOLOv10 weights.
3. **Text Extraction** – Detected plate regions passed to PaddleOCR for alphanumeric text recognition.
4. **Database & JSON Storage** – Recognized plates stored in SQLite (with start/end timestamps) and structured JSON logs.
5. **Visualization** – Bounding boxes and extracted text overlaid on frames using OpenCV.

## Model Training

- **Architecture:** YOLOv10-nano (transfer learning from pretrained YOLOv10 weights)
- **Epochs:** 120
- **Batch size:** 16
- **Data augmentation:** Scaling, flipping
- **Output weights:** `best.pt` (best-performing checkpoint)

### Performance

| Metric | Value |
|---|---|
| Precision | 90.1% |
| Recall | 79.4% |
| Overall Accuracy | ~84.7% |

## How It Works

1. The user selects **image** or **video** mode at runtime.
2. In video mode, frames are extracted sequentially using OpenCV; in image mode, the input image is loaded directly.
3. Each frame/image is passed to the YOLOv10 model, which localizes license plate regions with bounding boxes.
4. Detected regions are cropped and sent to PaddleOCR for text recognition.
5. Extracted text is cleaned/formatted to remove invalid symbols and correct common recognition errors.
6. Recognized plate numbers and timestamps are saved to the SQLite database and JSON files.
7. The processed output (bounding boxes + recognized text) is displayed in real time.

## Results

The system was evaluated on multiple test images and videos under varying lighting, angles, and background complexity:

- High detection accuracy with minimal false positives.
- Reliable text extraction even on partially blurred or slightly tilted plates.
- Real-time performance maintained during video processing.
- Correct and consistent storage of plate data in JSON and SQLite.

### Special Cases Observed

- **Full detection and content extraction** — clear plates correctly detected and read end-to-end, including under low-light/night conditions.
- **Partial detection and content extraction** — plates that are muddy, noisy, or at odd angles are sometimes only partially detected or read (e.g., a truncated or slightly incorrect character sequence).

## Comparison with Existing Systems

| Feature / Metric | This System (YOLOv10 + PaddleOCR) | Other Notable Systems (Literature) |
|---|---|---|
| Detection Model | YOLOv10-nano | YOLOv8, YOLOv5 |
| OCR Engine | PaddleOCR | EasyOCR, Tesseract, K-NN |
| Precision | 90.1% | Up to 100% (constrained datasets) |
| Recall | 79.4% | Up to 99.8% (constrained datasets) |
| Overall Accuracy | ~84.7% | Up to 98% (constrained datasets) |
| Key Strength | Balanced speed and accuracy, general-purpose | Specialized accuracy for specific plate types/conditions |
| Real-time Capability | Yes | Varies |

**Analysis:** Some literature systems report higher precision/recall, but they are typically tuned to narrow, constrained environments or specific plate formats (e.g., a single country's plates). This system aims for a generalized, real-time-capable solution with a strong balance between precision and speed.

## Limitations

1. **Partial detection in suboptimal conditions** — heavy dirt/mud, extreme motion blur, or highly oblique angles can reduce detection/recognition quality.
2. **Dependence on training data** — performance is tied to the diversity of the Roboflow training dataset; underrepresented plate formats, fonts, or severe weather conditions may reduce accuracy.
3. **Computational requirements** — a GPU is recommended for optimal real-time frame rates; low-power edge devices may need further optimization.
4. **OCR ambiguity errors** — visually similar characters (e.g., '0' vs 'O', '8' vs 'B') can occasionally be misclassified, especially in low-resolution or blurry crops.

## Future Work

- Flask-based web interface for direct uploads and real-time visualization in-browser.
- Multi-language support in PaddleOCR for broader regional applicability.
- Edge deployment optimization (e.g., Raspberry Pi, NVIDIA Jetson) for low-latency, on-site detection.
- Cloud synchronization of detection logs.
- Plate re-identification across frames/sessions.
- Advanced pre-processing for low-light and motion-blurred inputs.

## Tech Stack

- **Detection:** YOLOv10 (nano variant)
- **OCR:** PaddleOCR
- **Visualization:** OpenCV
- **Storage:** SQLite, JSON
- **Dataset Source:** Roboflow

## Project Structure (Suggested)

```
├── data/                  # Training/validation datasets (Roboflow export)
├── weights/
│   └── best.pt            # Trained YOLOv10 model weights
├── src/
│   ├── detect.py          # YOLOv10 detection logic
│   ├── ocr.py              # PaddleOCR text extraction logic
│   ├── storage.py          # SQLite + JSON storage handlers
│   └── main.py             # Entry point (image/video mode selection)
├── outputs/
│   ├── logs.json           # Cumulative detection logs
│   └── plates.db           # SQLite database
└── README.md
```

## Conclusion

This system combines YOLOv10 for license plate detection and PaddleOCR for text recognition into a robust, automated pipeline for vehicle identification. It achieves strong accuracy and real-time performance across varied environmental conditions, making it suitable for traffic surveillance, parking management, and automated access control, with SQLite and JSON logging ensuring systematic data storage and retrieval.
