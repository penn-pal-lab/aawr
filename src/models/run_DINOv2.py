from pathlib import Path
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dino_adapter import DINOV2Adapter

def main():
    image_path = "/mnt/sda/edward/data_mix_bookshelf_valid/success/2025-04-21/2025-04-21_15-21-04/recordings/frames/hand_camera/00032.jpg"
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    featureExtractor = DINOV2Adapter()
    feature = featureExtractor.extract(image)

    # Perform PCA
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(feature)  # Shape: [L, 3]

    # Construct a 2D feature map
    grid_size = int(np.sqrt(features_pca.shape[0]))
    features_pca = features_pca[:grid_size * grid_size]
    features_pca_image = features_pca.reshape(grid_size, grid_size, 3)

    # Normalize features to 0-255
    min_val = features_pca_image.min()
    max_val = features_pca_image.max()
    features_norm = (features_pca_image - min_val) / (max_val - min_val + 1e-5)
    features_uint8 = (features_norm * 255).astype(np.uint8)

    # Convert to PIL image
    result_img = Image.fromarray(features_uint8)

    # Post-processing
    upsample_size = image.size  # image.size returns (width, height)
    result_img_up = result_img.resize(upsample_size, resample=Image.BICUBIC)

    # Display
    plt.figure(figsize=(25, 12))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(result_img_up)
    plt.title("Extracted Feature + PCA")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


