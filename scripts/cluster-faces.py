#!/usr/bin/env python3
import argparse
import json
from src.clustering.cluster_faces import load_image_paths, compute_embeddings, cluster_embeddings, create_cluster_matrix_image

def main():
    parser = argparse.ArgumentParser(
        description="Cluster faces based on embeddings and output clustering results and a matrix image."
    )
    parser.add_argument("images_dir", help="Directory containing aligned (cropped) face images.")
    parser.add_argument("--output_json", default="identities.json", 
                        help="Path to output JSON file for clustering results.")
    parser.add_argument("--output_png", default="clustering_matrix.png", 
                        help="Path to output PNG matrix image.")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps parameter (for cosine metric).")
    parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN min_samples parameter.")
    parser.add_argument("--img_size", type=int, default=100, help="Size to which each face image is resized (square).")
    
    args = parser.parse_args()
    
    # 1. Load image paths.
    image_paths = load_image_paths(args.images_dir)
    print(f"Found {len(image_paths)} images in {args.images_dir}.")
    
    # 2. Compute embeddings.
    embeddings, filenames = compute_embeddings(image_paths)
    if len(embeddings) == 0:
        print("No embeddings were computed. Exiting.")
        return

    # 3. Cluster embeddings.
    labels = cluster_embeddings(embeddings, eps=args.eps, min_samples=args.min_samples)
    unique_labels = set(labels)
    print("Clustering completed. Unique labels found:", unique_labels)
    
    # 4. Create and save the matrix image.
    matrix_image, clusters = create_cluster_matrix_image(labels, filenames, image_size=(args.img_size, args.img_size))
    
    # Save clustering results as a JSON file.
    with open(args.output_json, "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"Clustering results saved to {args.output_json}")

    # Save the matrix image.
    if matrix_image is not None:
        import cv2
        cv2.imwrite(args.output_png, matrix_image)
        print(f"Clustering matrix image saved to {args.output_png}")
    else:
        print("Matrix image could not be created.")
    
if __name__ == "__main__":
    main()