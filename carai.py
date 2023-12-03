import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf


data_path_images = r'C:\Users\trild\PycharmProjects\CarAImodel\images'
data_path_labels = r'C:\Users\trild\PycharmProjects\CarAImodel\labels'

images = []
labels = []

# Загрузка изображений
for label, label_name in enumerate(os.listdir(data_path_images)):
    label_path_images = os.path.join(data_path_images, label_name)
    for img_file in os.listdir(label_path_images):
        img_path = os.path.join(label_path_images, img_file)

        image = cv2.imread(img_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {img_path}")
            continue

        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
        images.append(image)
        labels.append(label)

def process_yolo_label(label_info):
    lines = label_info.split('\n')
    labels = []

    for line in lines:
        data = line.split()
        if len(data) == 5:  # Ожидаемый формат данных: class x_center y_center width height
            class_label = int(data[0])
            x_center, y_center, width, height = map(float, data[1:])
            labels.append([class_label, x_center, y_center, width, height])

    return labels


# Загрузка разметок
for label, label_name in enumerate(os.listdir(data_path_labels)):
    label_path_labels = os.path.join(data_path_labels, label_name)
    for label_file in os.listdir(label_path_labels):
        label_file_path = os.path.join(label_path_labels, label_file)

        with open(label_file_path, 'r') as file:
            label_info = file.read().strip()
            processed_labels = process_yolo_label(label_info)

            # Преобразование списка меток в список списков меток для каждого изображения
            labels_per_image = []
            for label_set in processed_labels:
                labels_per_image.append(label_set)

            labels.append(labels_per_image)  # Добавляем список списков меток, полученных из функции process_yolo_label

labels = [str(sublist) for sublist in labels]

images = np.array(images)
labels = np.array(labels)

unique_images = []
unique_labels = []

# Проход по изображениям и меткам
for image, label_set in zip(images, labels):
    for label in label_set:
        unique_images.append(image)
        unique_labels.append(label)

# Преобразование списков в массивы numpy
unique_images = np.array(unique_images)
unique_labels = np.array(unique_labels)

# Разделение данных на обучающую, валидационную и тестовую выборки
train_images, test_images, train_labels, test_labels = train_test_split(unique_images, unique_labels, test_size=0.2, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.5, random_state=42)

# Создание последовательной модели
model = tf.keras.Sequential()

# Добавление слоев
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='softmax'))  # Выходной слой с количеством классов

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Вывод информации о модели
model.summary()