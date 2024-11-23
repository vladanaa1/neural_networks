import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import os
import numpy as np
import matplotlib.pyplot as plt

# DOWNLOAD DATA FROM: https://www.kaggle.com/datasets/morph1max/definition-of-cargo-transportation?resource=download-directory

main_path = './podaciCas06/'
folder_paths = [
'./podaciCas06/Background',
'./podaciCas06/Wood',
'./podaciCas06/Brick',
'./podaciCas06/Concrete',
'./podaciCas06/Ground'
]

num_files = [len(os.listdir(folder)) for folder in folder_paths]
folder_labels = [
    "Background",
    "Wood",
    "Brick",
    "Concrete",
    "Ground"
]

plt.figure(figsize=(10, 6))
plt.bar(range(len(folder_paths)), num_files, color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Number of Files')
plt.title('Number of Files in Each Class')
plt.xticks(range(len(folder_paths)), folder_labels)  # Set custom labels
plt.tight_layout()
plt.show()


def show_sample_images(folder_paths):
    plt.figure(figsize=(15, 10))
    for i, folder_path in enumerate(folder_paths):
        # Get a list of file names in the current folder
        files = os.listdir(folder_path)

        # Load and display the first image from the folder
        img_path = os.path.join(folder_path, files[0])
        img = plt.imread(img_path)

        # Plot the image
        plt.subplot(2, 3, i + 1)  # Adjust the subplot layout as needed
        plt.imshow(img)
        plt.title(folder_path.split('/')[-1])  # Extract class name from folder path
        plt.axis('off')
    plt.show()


# Call the function to display sample images
show_sample_images(folder_paths)
img_size = (64, 64)
batch_size = 64

from keras.utils import image_dataset_from_directory

Xtrain = image_dataset_from_directory(main_path,
                                      subset='training',
                                      validation_split=0.2,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      seed=123)

Xval = image_dataset_from_directory(main_path,
                                    subset='validation',
                                    validation_split=0.2,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123)

classes = Xtrain.class_names
print(classes)


from keras import layers
from keras import Sequential

data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(img_size[0],
                                                 img_size[1], 3)),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.1),
  ]
)


from keras import Sequential
from keras import layers
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy

num_classes = len(classes)
print(num_classes)
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')

history = model.fit(Xtrain,
                    epochs=50,
                    validation_data=Xval,
                    verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()
labels = np.array([])
pred = np.array([])
labels2 = np.array([])
pred2 = np.array([])
for img, lab in Xval:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))
for img, lab in Xtrain:
    labels2 = np.append(labels2, lab)
    pred2 = np.append(pred2, np.argmax(model.predict(img, verbose=0), axis=1))
from sklearn.metrics import accuracy_score
print('Tačnost modela je: ' + str(100*accuracy_score(labels, pred)) + '%')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()


cm2 = confusion_matrix(labels2, pred2, normalize='true')
cmDisplay2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=classes)
cmDisplay2.plot()
plt.show()