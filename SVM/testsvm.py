import cv2
import joblib
from skimage.feature import hog
import numpy as np

# Modeli yükle
model = joblib.load("flower_classifier.pkl")
print("Model yüklendi!")

# Resim boyutu ve sınıflar
IMG_SIZE = (128, 128)
CATEGORIES = ["Lilly", "Lotus", "Orchid", "Sunflower", "Tulip"]

def predict_flower(image_path):
    # Resmi yükle ve ön işle
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # HOG özelliklerini çıkar
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys',
                       transform_sqrt=True, visualize=False)
    
    # Tahmin yap
    hog_features = np.array([hog_features])  # Modelin tahmin edebilmesi için şekil (1, -1) olmalı
    prediction = model.predict(hog_features)
    
    # Sonucu yazdır
    predicted_class = CATEGORIES[prediction[0]]
    print(f"Tahmin edilen çiçek türü: {predicted_class}")

# Örnek kullanım
if __name__ == "__main__":
    test_image = "test1.jpg"  # Test etmek istediğiniz resim dosyasının yolu
    predict_flower(test_image)
