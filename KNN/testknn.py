import cv2
import joblib
import pywt
import numpy as np

# Model ve sınıflar
MODEL_PATH = "flower_knn_model.pkl"
IMG_SIZE = (128, 128)
CATEGORIES = ["lilly", "lotus", "orchid", "sunflower", "tulip"]

# DWT Özelliklerini Çıkarma
def extract_dwt_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coeffs2 = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL.flatten()

# Resim Üzerinden Tahmin Yapma
def predict_flower(image_path, model_path):
    # Modeli yükle
    print("Model yükleniyor...")
    model = joblib.load(model_path)
    print("Model yüklendi!")
    
    # Resmi işleme
    print(f"Resim işleniyor: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMG_SIZE)
    
    # DWT özelliklerini çıkarma
    features = extract_dwt_features(img).reshape(1, -1)  # Şekli (1, -1) olmalı
    
    # Tahmin yapma
    prediction = model.predict(features)
    predicted_class = CATEGORIES[prediction[0]]
    print(f"Tahmin edilen çiçek türü: {predicted_class}")

# Örnek Kullanım
if __name__ == "__main__":
    # Tahmin yapmak istediğiniz resmin yolu
    test_image_path = "test1.jpg"
    predict_flower(test_image_path, MODEL_PATH)
