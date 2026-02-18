import cv2
import json
import os
import re
import numpy as np
from datetime import datetime
from ultralytics import YOLOv10
from paddleocr import PaddleOCR

# Load image
image_path = "data/carImage5.jpg"
frame = cv2.imread(image_path)
if frame is None:
    print("Image not found or failed to load.")
    exit()

# Initialize models
model = YOLOv10("model/best.pt")
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Create output folder
os.makedirs("json", exist_ok=True)

# Detection
results = model.predict(frame, conf=0.25)
license_plates = set()

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = frame[y1:y2, x1:x2]

        # OCR
        ocr_result = ocr.ocr(cropped, det=False, rec=True, cls=False)
        text = ""
        for r in ocr_result:
            score = r[0][1]
            score = 0 if np.isnan(score) else int(score * 100)
            if score > 60:
                text = r[0][0]

        # Clean text
        text = re.sub(r'[\W]', '', text).replace(" ", "").replace(" ", " ").replace(" ", "")
        if text:
            license_plates.add(text)

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Save to JSON
start_time = datetime.now()
end_time = datetime.now()
interval_data = {
    "Start Time": start_time.isoformat(),
    "End Time": end_time.isoformat(),
    "License Plate": list(license_plates)
}

# Save individual file
filename = f"json/output_{start_time.strftime('%Y%m%d%H%M%S')}.json"
with open(filename, 'w') as f:
    json.dump(interval_data, f, indent=2)
print(f"Saved to {filename}")

# Save cumulative file
cumulative_path = "json/LicensePlateData.json"
if os.path.exists(cumulative_path):
    with open(cumulative_path, 'r') as f:
        existing_data = json.load(f)
else:
    existing_data = []

existing_data.append(interval_data)
with open(cumulative_path, 'w') as f:
    json.dump(existing_data, f, indent=2)
print(f"Updated cumulative file: {cumulative_path}")

# Show result
cv2.imshow("License Plate Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
