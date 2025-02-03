import argparse
import os
import glob
import json
import cv2
import numpy as np
from src.alignment.align_faces import align_face

def main():
    parser = argparse.ArgumentParser(description="Align faces in all images from a folder using detection results.")
    parser.add_argument("input_dir", help="Path to the folder containing the original images.")
    parser.add_argument("detection_json", help="Path to the JSON file with detection results (produced by detect-folder.py).")
    parser.add_argument("--output_json", default="alignment_results.json", 
                        help="Filename for the combined alignment results JSON file.")
    parser.add_argument("--processed_dir", default="processed", 
                        help="Directory where aligned face images will be saved.")
    args = parser.parse_args()

    # Create the processed folder if it doesn't exist
    os.makedirs(args.processed_dir, exist_ok=True)

    # Load the detection results
    with open(args.detection_json, "r") as f:
        detection_results = json.load(f)

    alignment_results = {}  # This will store updated info for each image

    # Process all images in the input directory
    image_patterns = [os.path.join(args.input_dir, "*.jpg"), os.path.join(args.input_dir, "*.png")]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(pattern))

    if not image_files:
        print("No image files found in the specified folder.")
        return

    for image_path in sorted(image_files):
        image_name = os.path.basename(image_path)
        if image_name not in detection_results:
            print(f"No detection results for {image_name}. Skipping.")
            continue

        print(f"Aligning faces for {image_name} ...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}. Skipping.")
            continue

        faces = detection_results[image_name]
        aligned_faces_info = []
        for i, face in enumerate(faces):
            try:
                aligned_face, affine_matrix = align_face(image, face["landmarks"])
            except Exception as e:
                print(f"Error aligning face {i} in {image_name}: {e}")
                continue

            # Define a new filename for the aligned image in the processed directory
            aligned_filename = f"{os.path.splitext(image_name)[0]}_aligned_{i}.jpg"
            aligned_path = os.path.join(args.processed_dir, aligned_filename)
            cv2.imwrite(aligned_path, aligned_face)

            # Update the face dictionary with transformation info
            face["transform"] = {
                "affine_matrix": affine_matrix,
                "aligned_image_path": aligned_path
            }
            aligned_faces_info.append(face)

        alignment_results[image_name] = aligned_faces_info

    # Save all alignment results in a single JSON file
    with open(args.output_json, "w") as f:
        json.dump(alignment_results, f, indent=2)

    print(f"Alignment results saved to {args.output_json}")

if __name__ == "__main__":
    main()
