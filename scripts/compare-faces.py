# compare-faces.py (updated to include distilled embeddings)
import argparse
import json
import os
import numpy as np
from src.utils.mse_utils import compute_mse
from src.embeddings.embedding_utils import PretrainedEmbeddingExtractor  # Your teacher extractor
from src.embeddings.distilled_embedding_utils import DistilledEmbeddingExtractor

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def main():
    parser = argparse.ArgumentParser(
        description="Compare faces using MSE, teacher embeddings, and distilled embeddings."
    )
    parser.add_argument("image1", help="Path to the first aligned face image (e.g., image1_aligned.jpg)")
    parser.add_argument("image2", help="Path to the second aligned face image (e.g., image2_aligned.jpg)")
    parser.add_argument("--output", default="comparison_results.json", help="Output JSON file for results")
    args = parser.parse_args()

    # Compute MSE (optionally scaled for readability)
    mse_score = compute_mse(args.image1, args.image2)
    scaled_mse = mse_score * 1e7

    # Teacher embedding extraction using your existing PretrainedEmbeddingExtractor.
    teacher_extractor = PretrainedEmbeddingExtractor()
    teacher_emb1 = np.array(teacher_extractor.extract_arcface_embedding(args.image1))
    teacher_emb2 = np.array(teacher_extractor.extract_arcface_embedding(args.image2))
    teacher_cosine = cosine_similarity(teacher_emb1, teacher_emb2)
    teacher_euclidean = euclidean_distance(teacher_emb1, teacher_emb2)

    # Distilled embedding extraction using the new extractor.
    distilled_extractor = DistilledEmbeddingExtractor()
    distilled_emb1 = distilled_extractor.extract_embedding(args.image1)
    distilled_emb2 = distilled_extractor.extract_embedding(args.image2)
    distilled_cosine = cosine_similarity(distilled_emb1, distilled_emb2)
    distilled_euclidean = euclidean_distance(distilled_emb1, distilled_emb2)

    # Prepare the output dictionary.
    results = {
        "MSE": round(scaled_mse, 4),
        "teacher_embedding": {
            "cosine_similarity": round(teacher_cosine, 4),
            "euclidean_distance": round(teacher_euclidean, 4)
        },
        "distilled_embedding": {
            "cosine_similarity": round(distilled_cosine, 4),
            "euclidean_distance": round(distilled_euclidean, 4)
        }
    }

    # Save the results to a JSON file.
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Comparison results saved to {args.output}")

if __name__ == "__main__":
    main()
