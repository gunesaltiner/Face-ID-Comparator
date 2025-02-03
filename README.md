# Face-ID-Comparator

Face-ID-Comparator is an end-to-end solution for automated face detection, alignment, comparison, and clustering. The project processes input images (which may include multiple faces) by performing the following steps:

- **Face Detection:** Detects faces and facial landmarks using the InsightFace framework.
- **Face Alignment:** Crops and aligns detected faces to a canonical coordinate system.
- **Face Comparison:** Compares faces using three methods:
  - Pixel-level Mean Squared Error (MSE)
  - Teacher (pretrained) face embeddings (e.g., ArcFace)
  - Distilled face embeddings from a lightweight model
- **Face Clustering:** Groups faces into clusters corresponding to unique individuals by applying DBSCAN clustering on the embedding space.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/gunesaltiner/Face-ID-Comparator.git
cd Face-ID-Comparator
```

### 2. Set Up a Conda Environment

Create and activate a Conda environment (named `myenv` with Python 3.9):

```bash
conda create -n myenv python=3.9
conda activate myenv
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Set the PYTHONPATH

Ensure that your project’s root (which contains the `src` package) is included in the Python module search path:

```bash
export PYTHONPATH=.
```

## Running the Scripts

### Face Detection

To process all images in the `data/images/` folder for face detection, run:

```bash
python scripts/detect-folder.py data/images/
```

This command processes each image, extracts face bounding boxes and landmarks using InsightFace, and writes the detection metadata to a JSON file (e.g., `detection_results.json`). Example output:

```
Processing data/images/sample1.jpg …
Processing data/images/sample12.jpg …
...
Detection results saved to detection_results.json
```

### Face Alignment

To align faces based on the detection metadata, run:

```bash
python scripts/align-folder.py data/images/ detection_results.json
```

Example output:

```
Aligning faces for sample1.jpg …
Aligning faces for sample12.jpg …
...
Alignment results saved to alignment_results.json
```

### Face Comparison

To compare two aligned face images using multiple methods (MSE, teacher embeddings, and distilled embeddings), run:

```bash
python scripts/compare-faces.py data/processed/sample1_aligned_1.jpg data/processed/sample2_aligned_3.jpg --output comparison_results.json
```

Example output:

```
Comparison results saved to comparison_results.json
```

### Face Clustering

To cluster all aligned face images (e.g., in the `data/processed/` folder), run:

```bash
python scripts/cluster-faces.py data/processed/ --output_json identities.json --output_png clustering_matrix.png --eps 0.5 --min_samples 2 --img_size 100
```

This command:
- Loads all aligned images.
- Computes embeddings (using the teacher extractor).
- Normalizes the embeddings and applies DBSCAN clustering (parameters `--eps` and `--min_samples` can be adjusted).
- Saves the clustering results as `identities.json`, mapping cluster labels to image paths.
- Creates a composite PNG image (`clustering_matrix.png`) where each row represents a cluster.

Example output:

```
Found 21 images in processed/.
Clustering completed. Unique labels found: {0, 1, -1}
Clustering results saved to identities.json
Clustering matrix image saved to clustering_matrix.png
```

## Overview of the Approach

### **Face Detection**
The detection module uses the InsightFace framework to locate faces and extract facial landmarks. The detection process produces a JSON file with bounding boxes, landmark coordinates, and detection scores.

### **Face Alignment**
Using the detected landmarks, the alignment module applies a 2D affine transformation to rotate and crop each face into a canonical size (typically 512×512 pixels). The aligned face images and updated metadata are saved for subsequent processing.

### **Face Comparison**
The comparison module offers multiple methods to assess similarity between faces:
- **MSE:** Computes pixel-level differences between aligned face images.
- **Teacher Embeddings:** Uses a pretrained model (e.g., ArcFace) to extract high-dimensional embeddings and computes similarity via cosine similarity and Euclidean distance.
- **Distilled Embeddings:** Employs a lightweight, distilled model trained to mimic the teacher embeddings, providing faster inference with comparable performance.

The results are compiled into a JSON file for further analysis.

### **Face Clustering**
The clustering module calculates embeddings for all aligned face images, normalizes these vectors, and applies DBSCAN clustering (using cosine similarity) to group images by identity. The output includes both a JSON file mapping cluster labels to image paths and a composite matrix image that visually represents the clusters.

---

**Author:** Gunes Altiner  
**Repository:** [Face-ID-Comparator](https://github.com/gunesaltiner/Face-ID-Comparator)
