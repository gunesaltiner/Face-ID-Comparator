import cv2
import numpy as np
import json
import argparse

# Reference landmarks in the 512x512 template (order => left_eye, right_eye, nose, mouth_left, mouth_right)
REF_LANDMARKS = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.1936],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118]
], dtype=np.float32)

def align_face(image, landmarks):
        
    #Extract detected landmarks
    detected_landmarks = np.array([
        [landmarks["left_eye"][0], landmarks["left_eye"][1]],
        [landmarks["right_eye"][0], landmarks["right_eye"][1]],
        [landmarks["nose"][0], landmarks["nose"][1]],
        [landmarks["mouth_left"][0], landmarks["mouth_left"][1]],
        [landmarks["mouth_right"][0], landmarks["mouth_right"][1]]
        ], dtype=np.float32)
        

    # get affine matrix with RANSAC    
    affine_matrix, _ = cv2.estimateAffine2D(detected_landmarks, REF_LANDMARKS, method=cv2.RANSAC)

    aligned_face = cv2.warpAffine(image, affine_matrix, (512,512), flags=cv2.INTER_CUBIC)

    return aligned_face, affine_matrix.tolist()

def main():
    parser = argparse.ArgumentParser(description="Align faces to 512x512.")
    parser.add_argument("image_path", help="Path to image path.")
    parser.add_argument("metadata_path", help="Path to metadata file (JSON format).")

    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    
    if image is None:
        raise FileNotFoundError(f"Image {args.image_path} is not found.")
    
    with open(args.metadata_path, "r") as f:
        metadata = json.load(f)

    # Align all detected face
    for i, face in enumerate(metadata["faces"]):
        # Align face
        aligned_face, affine_matrix = align_face(image, face["landmarks"])

        # Save aligned face image
        output_image_path = args.image_path.replace(".jpg", f"_aligned_{i}.jpg").replace(".png", f"_aligned_{i}.png")
        cv2.imwrite(output_image_path, aligned_face)

        # Add transformation matrix to JSON metadata
        face["transform"] = {
            "affine_matrix": affine_matrix,
            "aligned_image_path": output_image_path
        }

    with open(args.metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Aligned faces saved. Updated metadata at {args.metadata_path}")

if __name__ == "__main__":
    main()