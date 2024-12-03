import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import shutil
import cv2


def cluster_images(folder_path, n_clusters, output_path, batch_size=32, move_files=False):
    """
    Clusters images from the specified folder into n_clusters clusters and organizes them into subdirectories,
    sorting images within each cluster by their sharpness.

    :param folder_path: Path to the folder containing images.
    :param n_clusters: The number of clusters to form.
    :param output_path: Path to the output directory where clusters will be saved.
    :param batch_size: Number of images to process in a batch.
    :param move_files: If True, images will be moved instead of copied. Default is False.
    :return: A list of clusters, where each cluster is a list of tuples (image_path, sharpness).
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # List all image files in the directory
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(folder_path)
        for f in filenames
        if f.lower().endswith(valid_extensions)
    ]

    # Check if images are found
    if not image_files:
        raise ValueError("No images found in the specified folder.")

    # Sort the image files to maintain a consistent order
    image_files = sorted(image_files)

    # Convert the list of file paths to a TensorFlow dataset
    path_ds = tf.data.Dataset.from_tensor_slices(image_files)

    # Define a function to load and preprocess images
    def load_and_preprocess_image(path):
        # Read the image file
        img = tf.io.read_file(path)
        # Decode the image into a tensor
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        # Set shape for TensorFlow (unknown shape can cause issues)
        img.set_shape((None, None, 3))
        # Resize the image to the expected size
        img = tf.image.resize(img, [224, 224])
        # Preprocess the image for MobileNetV2
        img = preprocess_input(img)
        return img

    # Map the preprocessing function to the dataset
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch the dataset for optimal performance
    image_ds = image_ds.batch(batch_size)
    image_ds = image_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Load the pre-trained MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

    # Extract features from images in batches
    features = []
    for batch in image_ds:
        batch_features = base_model.predict(batch)
        features.append(batch_features)

    # Concatenate all the features
    features = np.concatenate(features, axis=0)

    # Reduce dimensionality if necessary
    pca = PCA(n_components=50, random_state=22)
    features_reduced = pca.fit_transform(features)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=22)
    kmeans.fit(features_reduced)
    labels = kmeans.labels_

    # Organize filenames and sharpness by cluster
    clusters = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        image_path = image_files[idx]
        # Calculate sharpness
        sharpness = calculate_sharpness(image_path)
        if sharpness is not None:
            clusters[label].append((image_path, sharpness))

    # Now sort images within each cluster by sharpness
    for cluster in clusters:
        # Sort in descending order of sharpness (higher sharpness first)
        cluster.sort(key=lambda x: x[1], reverse=True)

    # Create subdirectories for each cluster and copy/move images
    for idx, cluster in enumerate(clusters):
        # Define the path for the cluster subdirectory
        cluster_dir = os.path.join(output_path, f'cluster_{idx + 1}')
        os.makedirs(cluster_dir, exist_ok=True)

        for i, (img_path, sharpness) in enumerate(cluster):
            # Get the basename of the image file
            img_name = os.path.basename(img_path)
            # Optionally, prefix the filename with the index to maintain order
            img_name_with_index = f"{i + 1:03d}_{img_name}"
            dest_path = os.path.join(cluster_dir, img_name_with_index)

            # Copy or move the image file to the cluster directory
            try:
                if move_files:
                    shutil.move(img_path, dest_path)
                else:
                    shutil.copy2(img_path, dest_path)
            except Exception as e:
                print(f"Error while copying/moving {img_path} to {dest_path}: {e}")

        # Optionally, write a text file with sharpness values
        sharpness_file = os.path.join(cluster_dir, 'sharpness_values.txt')
        with open(sharpness_file, 'w') as f:
            for img_path, sharpness in cluster:
                f.write(f"{os.path.basename(img_path)}\t{sharpness}\n")

    return clusters


def calculate_sharpness(image_path):
    """
    Calculates the sharpness of an image using the variance of the Laplacian method.

    :param image_path: Path to the image file.
    :return: Sharpness value (variance of Laplacian) or None if an error occurs.
    """
    try:
        # Read the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Check if image is read correctly
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        # Compute the Laplacian of the image
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        # Compute the variance of the Laplacian
        variance = laplacian.var()
        return variance
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def main():
    folder_path = 'dataset'  # Input directory containing images
    n_clusters = 25  # Number of clusters you want
    output_path = 'output'  # Output directory for clusters
    batch_size = 128  # Adjust based on your system's memory
    move_files = False  # Set to True to move files instead of copying

    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    # Run the clustering and file organization
    clusters = cluster_images(folder_path, n_clusters, output_path, batch_size, move_files)

    # Print out the clusters (optional)
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx + 1}:")
        for filename in cluster:
            print(f"  {filename}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
