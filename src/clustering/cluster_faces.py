import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from src.embeddings.embedding_utils import PretrainedEmbeddingExtractor

def load_image_paths(images_dir, valid_extensions=('.jpg', '.jpeg', '.png')):

    image_paths = [
        os.path.join(images_dir, fname)
        for fname in os.listdir(images_dir)
        if fname.lower().endswith(valid_extensions)
    ]
    return image_paths

def compute_embeddings(image_paths):

    extractor = PretrainedEmbeddingExtractor()
    embeddings = []
    filenames = []
    for path in image_paths:
        try:
            emb = extractor.extract_arcface_embedding(path)
            embeddings.append(np.array(emb))
            filenames.append(path)
        except Exception as e:
            print(f"Warning: failed to extract embedding for {path}. Error: {e}")
    embeddings = np.array(embeddings)
    return embeddings, filenames

def cluster_embeddings(embeddings, eps=0.5, min_samples=2):

    # Normalize embeddings ==> cosine similarity is more meaningful.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-10)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings_norm)
    labels = clustering.labels_
    return labels

def create_cluster_matrix_image(labels, filenames, image_size=(100, 100), padding=5):

    clusters = {}
    for label, fname in zip(labels, filenames):
        clusters.setdefault(str(label), []).append(fname)

    row_images = []
    for cluster_id in sorted(clusters.keys(), key=lambda x: int(x)):
        imgs = []
        for fname in clusters[cluster_id]:
            img = cv2.imread(fname)
            if img is None:
                continue

            img = cv2.resize(img, image_size)
            imgs.append(img)
        if len(imgs) == 0:
            continue

        row = cv2.hconcat(imgs)

        row = cv2.copyMakeBorder(row, padding, padding, padding, padding,
                                 cv2.BORDER_CONSTANT, value=[255, 255, 255])
        row_images.append(row)

    if len(row_images) == 0:
        return None, clusters

    max_width = max(row.shape[1] for row in row_images)
    padded_rows = []
    for row in row_images:
        height, width = row.shape[:2]
        if width < max_width:
            pad_width = max_width - width

            row = cv2.copyMakeBorder(row, 0, 0, 0, pad_width,
                                     cv2.BORDER_CONSTANT, value=[255, 255, 255])
        padded_rows.append(row)

    matrix_image = cv2.vconcat(padded_rows)
    return matrix_image, clusters