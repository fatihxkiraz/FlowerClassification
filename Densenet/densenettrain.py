
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Parametreler
img_size = (256, 256)  # DenseNet201 için önerilen girdi boyutu
epochs = 20
batch_size = 32
learning_rate = 0.0005

# Google Drive'dan veri yükleme
from google.colab import drive
drive.mount('/content/drive')

data_dir_train = "/content/drive/My Drive/DATASET/FLOWERS/train"
data_dir_test = "/content/drive/My Drive/DATASET/FLOWERS/test"

# Verisetini hazırlama
datagen_train = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalizasyon
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

datagen_test = ImageDataGenerator(rescale=1.0 / 255)  # Sadece normalizasyon

train_generator = datagen_train.flow_from_directory(
    data_dir_train,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = datagen_test.flow_from_directory(
    data_dir_test,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# DenseNet201 tabanını yükleme
base_model = DenseNet201(weights="imagenet", include_top=False, input_shape=(*img_size, 3))

# Modelin üst katmanlarını oluşturma
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Aşırı öğrenmeyi önlemek için Dropout
dense_layer = Dense(train_generator.num_classes, activation="softmax")(x)

# Modeli oluşturma
model = Model(inputs=base_model.input, outputs=dense_layer)

# Taban modeli dondur
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleme
model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

# Modeli eğitme
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
)

# Sonuçları görselleştirme
plt.figure(figsize=(12, 6))

# Eğitim ve doğrulama kaybı
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Eğitim ve doğrulama doğruluğu
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Modeli değerlendirme
eval_results = model.evaluate(test_generator)
print(f"Test Loss: {eval_results[0]:.4f}")
print(f"Test Accuracy: {eval_results[1]:.4f}")

# Modeli kaydetme
model.save("flower_classification_densenet201.h5")

# Test Generator: shuffle=False ayarı
test_generator = datagen_test.flow_from_directory(
    data_dir_test,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False  # Sıralama sabit kalmalı
)

# Tahminleri al
predictions = model.predict(test_generator, verbose=1)

# Tahmin edilen sınıflar
predicted_classes = np.argmax(predictions, axis=1)  # En yüksek olasılıklı sınıf

# Gerçek sınıflar
true_classes = test_generator.classes  # Test setindeki gerçek sınıflar

# Sınıflandırma raporu için sınıf etiketleri
class_labels = list(test_generator.class_indices.keys())

# Doğruluk kontrolü
if len(true_classes) == len(predicted_classes):  # Uzunlukların eşleştiğinden emin olun
    print(f"Test Accuracy: {accuracy_score(true_classes, predicted_classes):.4f}")
else:
    print("Hata: Gerçek sınıflar ile tahmin edilen sınıflar uzunlukları uyuşmuyor.")

# Sınıflandırma raporu oluştur
report = classification_report(true_classes, predicted_classes, target_names=class_labels, digits=2)
print("Sınıflandırma Raporu:")
print(report)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Gerçek sınıflar ve tahmin edilen sınıflar
true_classes = test_generator.classes
predicted_classes = np.argmax(predictions, axis=1)

# Sınıf isimleri
class_labels = list(test_generator.class_indices.keys())

# Karışıklık matrisi oluşturma
cm = confusion_matrix(true_classes, predicted_classes)

# Karışıklık matrisini görselleştirme
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues, values_format='d')  # Matris değerleri tam sayı olarak gösterilir
plt.title("Confusion Matrix")
plt.xticks(rotation=45)  # Sınıf isimlerini döndürmek için
plt.show()