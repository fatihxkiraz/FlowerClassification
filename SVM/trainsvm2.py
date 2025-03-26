import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# input yerleri
TRAIN_DIR = "dataset/train"  
TEST_DIR = "dataset/test"    
CATEGORIES = ["lilly", "lotus", "orchid", "sunflower", "tulip"]  
IMG_SIZE = (128, 128) 

# input etme + HOG
def load_data_and_extract_features(data_dir, categories):
    features = []
    labels = []
    for label, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, IMG_SIZE)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # HOG özellikleri çıkarma
                hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2), block_norm='L2-Hys',
                                   transform_sqrt=True, visualize=False)
                
                features.append(hog_features)
                labels.append(label)
            except Exception as e:
                print("image error:", e)
    return np.array(features), np.array(labels)

# main
if __name__ == "__main__":
    print("Eğitim verisi yükleme ve HOG özellikleri çıkartma")
    X_train, y_train = load_data_and_extract_features(TRAIN_DIR, CATEGORIES)
    print("Eğitim verisi yüklendi")
    
    print("Test verisi yükleme ve HOG özellikleri çıkartma")
    X_test, y_test = load_data_and_extract_features(TEST_DIR, CATEGORIES)
    print("Test verisi yüklendi")
    
    # Bir örneğin HOG özellik vektörünü yazdır
    print("\nBirinci görüntü için HOG özellik vektörü:")
    print(X_train[0])  # İlk eğitim örneğinin HOG özellik vektörünü yazdır
    print("\nHOG özellik vektörünün boyutu:")
    print(len(X_train[0]))  # HOG vektörünün uzunluğunu yazdır
    
    # SVM modeli eğitme
    print("Model eğitiliyor")
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)
    print("Model eğitimi tamamlandı")
    
    # Test etme
    y_pred = model.predict(X_test)
    
    # Model performansı
    acc = accuracy_score(y_test, y_pred)
    print("Test doğruluğu: {:.2f}%".format(acc * 100))
    print("Sınıflandırma raporu:\n", classification_report(y_test, y_pred, target_names=CATEGORIES))
    
    # Confusion matrix oluşturma
    print("Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Confusion matrix görselleştirme
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Confusion Matrix")
    plt.show()

import joblib

# Model kaydetme
joblib.dump(model, "svm2.pkl")
print("Model kaydedildi")
