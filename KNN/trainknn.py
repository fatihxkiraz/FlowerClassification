import os
import cv2
import numpy as np
import pywt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# veri yerleri
DATA_DIR = "dataset" 
TRAIN_DIR = os.path.join(DATA_DIR, "train")  # eğitim verisi yolu
TEST_DIR = os.path.join(DATA_DIR, "test")    # test verisi yolu
CATEGORIES = ["lilly", "lotus", "orchid", "sunflower", "tulip"]
IMG_SIZE = (256, 256)  # resim boyutu 

# dwt özellikleri
def extract_dwt_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coeffs2 = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL.flatten()

# veriyi yükleme ve dwt özelliklerini çıkarma
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
                
                # dwt özellikleri
                dwt_features = extract_dwt_features(img)
                features.append(dwt_features)
                labels.append(label)
            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
    return np.array(features), np.array(labels)

# main
if __name__ == "__main__":
    # eğitim verisi yükleme
    print("Eğitim verisi yükleme ve DWT özellikleri çıkartma")
    X_train, y_train = load_data_and_extract_features(TRAIN_DIR, CATEGORIES)
    print(f"Eğitim verisi yüklendi. Toplam {len(X_train)} örnek")

    # test verisi yükleme
    print("Test verisi yükleme ve DWT özellikleri çıkartma")
    X_test, y_test = load_data_and_extract_features(TEST_DIR, CATEGORIES)
    print(f"Test verisi yüklendi. Toplam {len(X_test)} örnek")
    
    # kNN Eğitme
    print("kNN modeli eğitiliyor...")
    knn_model = KNeighborsClassifier(n_neighbors=2)
    knn_model.fit(X_train, y_train)
    print("Model eğitimi tamamlandı")
    
    # Test 
    y_pred = knn_model.predict(X_test)
    
    # Performans değerlendirme
    acc = accuracy_score(y_test, y_pred)
    print("Test doğruluğu: {:.2f}%".format(acc * 100))
    print("Sınıflandırma raporu:\n", classification_report(y_test, y_pred, target_names=CATEGORIES))
    
    # Confusion Matrix oluşturma ve görselleştirme
    print("Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Görselleştirme
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.title("Confusion Matrix")
    plt.show()
    
    # Modeli kaydetme
    joblib.dump(knn_model, "flower_knn_model.pkl")
    print("Model kaydedildi")
