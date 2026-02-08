
# Import libraries
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


train_dir = "Dataset/Training"
test_dir = "Dataset/Testing"

classes = os.listdir(train_dir)

print("Classes:", classes)

"""Dataset Exploration (EDA)"""

# Count images in each class
image_count = {}
for class_name in classes:
  class_path = os.path.join(train_dir, class_name)
  image_count[class_name] = len(os.listdir(class_path))

print(image_count)

# Plot Class distribution
plt.figure(figsize=(10, 6))
plt.bar(image_count.keys(), image_count.values())
plt.xlabel('Tumor Classes')
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.show()

#Helps justify data augmentation
#Shows awareness of imbalance

# Display Sample MRI Images
plt.figure(figsize=(8, 8))
for i, class_name in enumerate(classes):
  class_path = os.path.join(train_dir, class_name)
  img_name = os.listdir(class_path)[0]
  img_path = os.path.join(class_path, img_name)

  img = cv2.imread(img_path)
  # Removed incorrect cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
  # Added cmap='gray' for proper grayscale display
  plt.subplot(2, 2, i+1)
  plt.imshow(img, cmap='gray')
  plt.title(class_name)
  plt.axis('off')

plt.show()

# Check image shape & channels
img.shape

# Check pixel value range
print("Min pixel value:", img.min())
print("Max pixel value:", img.max())

"""**Image Preprocessing & Augmentation**
 are performed to:

Make all images uniform in size

Improve model convergence

Reduce overfitting

Increase model generalization
"""

# Import required libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define Image Parameters
IMG_SIZE = 224
BATCH_SIZE = 32

# Creating ImageDataGenerator for Training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range = 0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Create Training Data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Create validation Data generator
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Create Test data generator (No augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Verify class labels
print(train_generator.class_indices)

# Visualize Augmented Images
images, labels = next(train_generator)

plt.figure(figsize=(10, 10))
for i in range(9):
  plt.subplot(3, 3, i+1)
  plt.imshow(images[i])
  plt.axis('off')
plt.show()

"""**Model Selection & Architecture**"""

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Load pretrained Base Models
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Add custom classification Head (4 neurons --> 4 classes)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

# Final model create
model = Model(inputs=base_model.input, outputs=output)     #functional API model class in keras,  allows tomodify existing models

# compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

"""**Model Training & Validation**

Objective of This Step

Train the CNN on MRI images

Monitor performance on validation data

Avoid overfitting

Save the best model
"""

# Importing Training Utilities
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define callbacks
early_stop = EarlyStopping(           # stops training when validation loss stops improving, prevents overfitting
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Model checklist (saves best performing model)
checkpoint = ModelCheckpoint(
    'brain_tumor_resnet50.h5',
    monitor='val_accuracy',
    save_best_only=True
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)

# Plot training & Validation Curves
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

"""**Model Evaluation & Performance Metrics**"""

# Load best saved model
from tensorflow.keras.models import load_model

model = load_model("brain_tumor_resnet50.h5")

# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)

# Generate Predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# True labels
y_true = test_generator.classes

from sklearn.metrics import classification_report

class_names = list(test_generator.class_indices.keys())
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Load & Preprocess Image

img_path = "Dataset/Testing/glioma/Te-gl_0213.jpg"   # Path to an actual MRI image from the dataset
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
predicted_class = np.argmax(prediction)

class_labels = list(test_generator.class_indices.keys())
print("Predicted Tumor Type:", class_labels[predicted_class])

model.save("brain_tumor_resnet50.h5")
