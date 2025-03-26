
#kütüphaneler
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
from google.colab import drive
import signal
import matplotlib.pyplot as plt

#timeout mekanizması
def timeout_handler(signum, frame):
    raise TimeoutError

signal.signal(signal.SIGALRM, timeout_handler)

# Google Drive'ı bağlayın
drive.mount('/content/drive')

# Veri yollarını belirleyin
train_dir = '/content/drive/My Drive/DATASET/FLOWERS/train'
test_dir = '/content/drive/My Drive/DATASET/FLOWERS/test'

#özellik çıkarma ve etiketleme fonksiyonu
def extract_features_and_labels_safe(data_dir):
    labels = []
    features = []
    sift = cv2.SIFT_create()

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(class_path, img_name)
            try:
                signal.alarm(5)  #5 saniye sınır
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Skipped: {img_path} (could not read image)")
                    continue

                #resize
                img = cv2.resize(img, (256, 256))

                keypoints, descriptors = sift.detectAndCompute(img, None)
                signal.alarm(0)  # alarm kapat

                if descriptors is not None:
                    descriptors = np.mean(descriptors, axis=0)  # özellik vektörleri ortalaması
                    features.append(descriptors)
                    labels.append(class_name)
                else:
                    print(f"Skipped: {img_path} (no SIFT features)")
            except TimeoutError:
                print(f"Timeout: {img_path}")
                continue
    return np.array(features), np.array(labels)

# Eğitim ve test özelliklerini ve etiketlerini çıkarın
print("eğitim verilerinden özellik çıkarılıyor")
X_train, y_train = extract_features_and_labels_safe(train_dir)
print("test verilerinden özellik çıkarılıyor")
X_test, y_test = extract_features_and_labels_safe(test_dir)

#etiketleri sayısal değerlere dönüştürme
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

#model eğitme
print("Model eğitiliyor...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

#tahmin
y_pred = rf.predict(X_test)

#sonuçları değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f"Model doğruluk oranı: {accuracy:.2f}")

# Confusion Matrix Analizi

cm = confusion_matrix(y_test, y_pred)
display_labels = le.classes_
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
cmd.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

# Precision, Recall ve F1-Score Analizi
print("\nSınıf bazında performans metrikleri:")
report = classification_report(y_test, y_pred, target_names=display_labels)
print(report)

"""# Yeni Bölüm"""