import cv2
import json
import numpy as np

# Load image and JSON
image_path = "aa.jpg"
json_path = "aa.json"
output_path = "aa_landmarks2.jpg"  # Output image with landmarks

image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image {image_path} not found.")

# Load face detection results
with open(json_path, "r") as f:
    data = json.load(f)

# Draw bounding boxes and landmarks
for face in data["faces"]:
    bbox = face["bbox"]  # [x, y, width, height]
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Draw bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

    # Draw landmarks
    for key, landmark in face["landmarks"].items():
        lx, ly = int(landmark[0]), int(landmark[1])
        cv2.circle(image, (lx, ly), 3, (0, 0, 255), -1)  # Red dot
        cv2.putText(image, key, (lx + 5, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Blue text

# Save the output image
cv2.imwrite(output_path, image)
print(f"Landmark visualization saved as {output_path}")

# Display the image (optional)
cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
