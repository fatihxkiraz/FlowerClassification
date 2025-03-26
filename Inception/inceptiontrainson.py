
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Parametreler
img_size = (256, 256)  # boyut
epochs = 20
batch_size = 64  # 128den
learning_rate = 0.001

# Google Drive'dan veri yükleme
from google.colab import drive
drive.mount('/content/drive')

data_dir_train = "/content/drive/My Drive/DATASET/FLOWERS/train"
data_dir_test = "/content/drive/My Drive/DATASET/FLOWERS/test"

#verisetini hazırlama
datagen_train = ImageDataGenerator(
    rescale=1.0 / 255,  # normalizasyon
    horizontal_flip=True,
    rotation_range=30,
    zoom_range=0.2
)

datagen_test = ImageDataGenerator(rescale=1.0 / 255)

train_generator = datagen_train.flow_from_directory(
    data_dir_train,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

test_generator = datagen_test.flow_from_directory(
    data_dir_test,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# InceptionV3 tabanını yükleme
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(*img_size, 3))

# modelin üst katmanları
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Aşırı öğrenmeyi önlemek için Dropout
dense_layer = Dense(train_generator.num_classes, activation="softmax")(x)

# modeli oluşturma
model = Model(inputs=base_model.input, outputs=dense_layer)

# taban modeli dondur
for layer in base_model.layers[:249]:  # İlk 249 katmanı dondur
    layer.trainable = False
for layer in base_model.layers[249:]:  # Son katmanları eğit
    layer.trainable = True

# modeli derleme
model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

# erken durdurmak için callback'i
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# modeli eğitme
try:
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[early_stopping],
        verbose=1
    )
except Exception as e:
    print(f"eğitim error: {e}")



# Sonuçları görselleştirme
if 'history' in locals():
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

#modeli değerlendirme
if 'history' in locals():
    eval_results = model.evaluate(test_generator)
    print(f"Test Loss: {eval_results[0]:.4f}")
    print(f"Test Accuracy: {eval_results[1]:.4f}")

#modeli kaydetme
if 'history' in locals():
    model.save("flower_classification_inceptionv3.h5")

from sklearn.metrics import classification_report
import numpy as np

#test verilerinin tahminleirni alma
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

#sınıflara göre rapor
report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels,
    digits=2
)

print("Sınıflandırma Raporu:")
print(report)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Test setinden tahminler
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)  # Tahmin edilen sınıflar
true_classes = test_generator.classes  # Gerçek sınıflar
class_labels = list(test_generator.class_indices.keys())  # Sınıf isimleri

# Confusion Matrix oluşturma
cm = confusion_matrix(true_classes, predicted_classes)

# Confusion Matrix'i görselleştirme
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()