import argparse
import os
import glob
import json
from src.detector.face_detector import FaceDetector

def main():
    parser = argparse.ArgumentParser(description="Perform face detection on all images in a folder.")
    parser.add_argument("input_dir", help="Path to the folder containing images.")
    parser.add_argument("--output_json", default="detection_results.json", 
                        help="Filename for the combined detection JSON file.")
    args = parser.parse_args()

    # Initialize your face detector
    detector = FaceDetector()

    detection_results = {}  # This dict will map image filename to its list of detected faces

    # Process all JPG and PNG images in the input directory
    image_patterns = [os.path.join(args.input_dir, "*.jpg"), os.path.join(args.input_dir, "*.png")]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(pattern))

    if not image_files:
        print("No image files found in the specified folder.")
        return

    for image_path in sorted(image_files):
        print(f"Processing {image_path} ...")
        try:
            faces = detector.detect_faces(image_path)
            # Use the basename (e.g., Aaron_Eckhart_0001.jpg) as the key
            image_name = os.path.basename(image_path)
            detection_results[image_name] = faces
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Save all detection results in a single JSON file
    with open(args.output_json, "w") as f:
        json.dump(detection_results, f, indent=2)

    print(f"Detection results saved to {args.output_json}")

if __name__ == "__main__":
    main()
