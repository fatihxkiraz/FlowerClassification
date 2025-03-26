import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Function to load images and extract labels
def load_images_and_labels(folder_path):
    images = []
    labels = []
    label_map = {}
    for label_index, label_name in enumerate(os.listdir(folder_path)):
        label_map[label_index] = label_name
        label_folder = os.path.join(folder_path, label_name)
        for image_name in os.listdir(label_folder):
            image_path = os.path.join(label_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
                labels.append(label_index)
    return images, labels, label_map

# Function to extract SIFT features
def extract_sift_features(images):
    sift = cv2.SIFT_create()
    descriptor_list = []
    for i, image in enumerate(images):
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            descriptor_list.append(descriptors)
        else:
            print(f"No descriptors found for image {i}")
            descriptor_list.append(None)
    return descriptor_list

# Function to create feature vectors (with histogram for each image)
def create_feature_vectors(descriptor_list, k):
    print("Clustering descriptors...")
    bow_trainer = cv2.BOWKMeansTrainer(k)
    for d in descriptor_list:
        if d is not None and len(d) > 0:
            bow_trainer.add(np.array(d))
    
    try:
        cluster_centers = bow_trainer.cluster()
        print("Cluster centers created.")
    except Exception as e:
        print(f"Error during clustering: {e}")
        return None

    histograms = []
    for i, descriptors in enumerate(descriptor_list):
        if descriptors is None or len(descriptors) == 0:
            print(f"Skipping empty descriptor for image {i}")
            histograms.append(np.zeros(k, dtype=np.float32))
        else:
            histogram = np.zeros(k, dtype=np.float32)
            # Assign each descriptor to the closest cluster
            cluster_assignments = [np.argmin(np.linalg.norm(cluster_centers - x, axis=1)) for x in descriptors]
            histogram += np.bincount(cluster_assignments, minlength=k)
            histograms.append(histogram)
    return np.array(histograms)

# Main script
train_folder = "dataset/train"
test_folder = "dataset/test"

if not os.path.exists(train_folder):
    print("Train folder does not exist.")
    exit()

if not os.path.exists(test_folder):
    print("Test folder does not exist.")
    exit()

# Load images and labels
print("Loading training images...")
train_images, train_labels, label_map = load_images_and_labels(train_folder)
print("Loading test images...")
test_images, test_labels, _ = load_images_and_labels(test_folder)

# Extract SIFT features
print("Extracting SIFT features from training images...")
train_descriptors = extract_sift_features(train_images)

# Create feature vectors for the training set
k = 100  # Number of clusters
train_features = create_feature_vectors(train_descriptors, k)
if train_features is None:
    print("Failed to create feature vectors for training set.")
    exit()

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(train_features, train_labels)
print("Random Forest training completed.")

# Extract SIFT features from test set
print("Extracting SIFT features from test images...")
test_descriptors = extract_sift_features(test_images)

# Create feature vectors for the test set
test_features = create_feature_vectors(test_descriptors, k)
if test_features is None:
    print("Failed to create feature vectors for test set.")
    exit()

# Predict and evaluate
print("Evaluating the model on the test set...")
predictions = rf_model.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(test_labels, predictions, target_names=[label_map[i] for i in range(len(label_map))]))
