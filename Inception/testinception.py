import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# 1. Modeli Yükleme
from tensorflow.keras.models import load_model

model_path = "flower_classification_inceptionv3.h5"
model = load_model(model_path, compile=False)

print("Model başarıyla yüklendi!")

# 2. Test Veri Kümesini Hazırlama
test_dir = "dataset/test"  # Test verisinin bulunduğu klasör
img_size = (128, 128)  # InceptionV3 için giriş boyutu
batch_size = 128

# Test verilerini hazırlama
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False  # Confusion Matrix için gerekli
)

# 3. Modeli Test Etme
print("Model test verileri üzerinde tahmin yapıyor...")
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# 4. Performans Analizi
class_labels = list(test_generator.class_indices.keys())
print("\nSınıflandırma Raporu:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# 5. Confusion Matrix Görüntüleme
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
