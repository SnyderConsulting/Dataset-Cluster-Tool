import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def cluster_images(folder_path, n_clusters):
    """
    Clusters images from the specified folder into n_clusters clusters.

    :param folder_path: Path to the folder containing images.
    :param n_clusters: The number of clusters to form.
    :return: A list of clusters, where each cluster is a list of image filenames.
    """
    # Get list of image files in the folder
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(valid_extensions)
    ]

    # Check if images are found
    if not image_files:
        raise ValueError("No images found in the specified folder.")

    # Load pre-trained VGG16 model + higher level layers
    model = VGG16(weights='imagenet', include_top=False)

    # Lists to hold features and corresponding image filenames
    features = []
    img_filenames = []

    # Iterate over images and extract features
    for img_path in image_files:
        try:
            # Load image and resize to expected size for VGG16
            img = image.load_img(img_path, target_size=(224, 224))
            # Convert image to array
            img_data = image.img_to_array(img)
            # Expand dimensions to match the model's input format
            img_data = np.expand_dims(img_data, axis=0)
            # Preprocess the image data
            img_data = preprocess_input(img_data)
            # Extract features
            feature = model.predict(img_data)
            # Flatten the features
            features.append(feature.flatten())
            # Save the image filename
            img_filenames.append(img_path)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    # Convert features to a numpy array
    features = np.array(features)

    # Optional: Reduce dimensionality to speed up clustering
    pca = PCA(n_components=50, random_state=22)
    features_reduced = pca.fit_transform(features)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=22)
    kmeans.fit(features_reduced)
    labels = kmeans.labels_

    plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=labels)
    plt.show()

    # Organize filenames by cluster
    clusters = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].append(img_filenames[idx])

    return clusters


def main():
    cluster_images("dataset", 5)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
