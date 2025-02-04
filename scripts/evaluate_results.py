import os
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from src.utils.mse_utils import compute_mse
from src.embeddings.embedding_utils import PretrainedEmbeddingExtractor
from src.embeddings.distilled_embedding_utils import DistilledEmbeddingExtractor

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def save_table_as_image(table_data, headers, filename):

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Evaluation table saved as {filename}")

def main():
    
    processed_folder = "processed/"
    image_files = [os.path.join(processed_folder, f)
                   for f in os.listdir(processed_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) < 2:
        print("Not enough images in the processed folder to compute pairwise metrics.")
        return

    mse_scores = []
    teacher_cosines = []
    teacher_euclidean = []
    distilled_cosines = []
    distilled_euclidean = []

    teacher_extractor = PretrainedEmbeddingExtractor()
    distilled_extractor = DistilledEmbeddingExtractor()

    teacher_embeddings = {}
    distilled_embeddings = {}
    for img_path in image_files:
        try:
            teacher_embeddings[img_path] = np.array(teacher_extractor.extract_arcface_embedding(img_path))
        except Exception as e:
            print(f"Error extracting teacher embedding for {img_path}: {e}")
        try:
            distilled_embeddings[img_path] = np.array(distilled_extractor.extract_embedding(img_path))
        except Exception as e:
            print(f"Error extracting distilled embedding for {img_path}: {e}")

    for img1, img2 in itertools.combinations(image_files, 2):
        try:
            mse_val = compute_mse(img1, img2)
            mse_val = mse_val * 1e7
            mse_scores.append(mse_val)
        except Exception as e:
            print(f"Error computing MSE for {img1} and {img2}: {e}")

        if img1 in teacher_embeddings and img2 in teacher_embeddings:
            emb1_teacher = teacher_embeddings[img1]
            emb2_teacher = teacher_embeddings[img2]
            teacher_cos = cosine_similarity(emb1_teacher, emb2_teacher)
            teacher_euc = np.linalg.norm(emb1_teacher - emb2_teacher)
            teacher_cosines.append(teacher_cos)
            teacher_euclidean.append(teacher_euc)

        if img1 in distilled_embeddings and img2 in distilled_embeddings:
            emb1_distilled = distilled_embeddings[img1]
            emb2_distilled = distilled_embeddings[img2]
            distilled_cos = cosine_similarity(emb1_distilled, emb2_distilled)
            distilled_euc = np.linalg.norm(emb1_distilled - emb2_distilled)
            distilled_cosines.append(distilled_cos)
            distilled_euclidean.append(distilled_euc)

    mse_mean = np.mean(mse_scores) if mse_scores else float('nan')
    mse_std = np.std(mse_scores) if mse_scores else float('nan')
    teacher_cos_mean = np.mean(teacher_cosines) if teacher_cosines else float('nan')
    teacher_euclid_mean = np.mean(teacher_euclidean) if teacher_euclidean else float('nan')
    distilled_cos_mean = np.mean(distilled_cosines) if distilled_cosines else float('nan')
    distilled_euclid_mean = np.mean(distilled_euclidean) if distilled_euclidean else float('nan')

    teacher_times = []
    distilled_times = []
    for img_path in image_files:
        start = time.time()
        try:
            _ = teacher_extractor.extract_arcface_embedding(img_path)
        except Exception as e:
            print(f"Error during teacher inference for {img_path}: {e}")
        teacher_times.append(time.time() - start)

        start = time.time()
        try:
            _ = distilled_extractor.extract_embedding(img_path)
        except Exception as e:
            print(f"Error during distilled inference for {img_path}: {e}")
        distilled_times.append(time.time() - start)

    teacher_avg_time = np.mean(teacher_times) if teacher_times else float('nan')
    distilled_avg_time = np.mean(distilled_times) if distilled_times else float('nan')

    table_data = [
        ["MSE (mean ± std)", f"{mse_mean:.4f} ± {mse_std:.4f}"],
        ["Teacher Cosine Similarity (mean)", f"{teacher_cos_mean:.4f}"],
        ["Teacher Euclidean Distance (mean)", f"{teacher_euclid_mean:.4f}"],
        ["Distilled Cosine Similarity (mean)", f"{distilled_cos_mean:.4f}"],
        ["Distilled Euclidean Distance (mean)", f"{distilled_euclid_mean:.4f}"],
        ["Teacher Model Avg. Inference Time (s)", f"{teacher_avg_time:.4f}"],
        ["Distilled Model Avg. Inference Time (s)", f"{distilled_avg_time:.4f}"],
    ]

    try:
        print("\nEvaluation Metrics:")
        print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
    except ImportError:
        print("\nEvaluation Metrics:")
        for row in table_data:
            print(f"{row[0]:40s} : {row[1]}")

    save_table_as_image(table_data, headers=["Metric", "Value"], filename="evaluation_table.png")

if __name__ == "__main__":
    main()