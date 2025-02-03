import argparse
import json
from src.detector.face_detector import FaceDetector

def main():
    parser = argparse.ArgumentParser(description='RetinaFace is used for Face Detection.')
    parser.add_argument('image_path', type=str, help='Path to input image.')
    args = parser.parse_args()

    # Detect faces
    detector = FaceDetector()
    faces = detector.detect_faces(args.image_path)

    # Save metadata to JSON
    output_path = args.image_path.replace('.png', '.json').replace('.jpg', '.json')
    with open(output_path, 'w') as f:
        json.dump({"faces": faces}, f, indent=2)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()