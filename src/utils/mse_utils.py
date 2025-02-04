import cv2
import numpy as np

def compute_mse(image_path1: str, image_path2: str) -> float:

    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    if image1 is None:
        print(f"Error: {image_path1} failed to load!")
    if image2 is None:
        print(f"Error: {image_path2} failed to load!")
    if image1.shape != image2.shape:
        raise ValueError("Shapes of images must be same.")
    
    image1 = image1.astype(np.float32) / 255.0
    image2 = image2.astype(np.float32) / 255.0

    squared_diff = np.square(image1 - image2)

    rows, cols, _ = image1.shape
    sigma = min(rows, cols) / 4  # Adjust sigma to control weighting
    mask = _create_gaussian_mask(rows, cols, sigma)
    squared_diff *= mask[:, :, np.newaxis]

    return float(np.mean(squared_diff))

def _create_gaussian_mask(rows: int, cols: int, sigma: float) -> np.ndarray:
    """Generate a 2D Gaussian mask centered on the image."""
    x = np.linspace(0, cols-1, cols)
    y = np.linspace(0, rows-1, rows)
    x_grid, y_grid = np.meshgrid(x, y)
    x_center, y_center = cols // 2, rows // 2
    mask = np.exp(-((x_grid - x_center)**2 + (y_grid - y_center)**2) / (2 * sigma**2))
    mask /= mask.sum()  # Normalize to sum=1
    return mask


