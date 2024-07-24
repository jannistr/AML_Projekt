# Diese Python-Datei wurde verwendet, um das eigene Modell mit dem eigenen Datensatz zu trainieren.

# Bibliotheken installieren
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers

# Datensatz in Trainings- und Validierungsdaten aufteilen (80/20) und Data-Flow als Vorbereitung fürs Training erstellen
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    r'C:\Users\jtrapp\OneDrive - Deloitte (O365D)\Documents\Studium\6. Semester\AML-Projekt\eigener Datensatz',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    r'C:\Users\jtrapp\OneDrive - Deloitte (O365D)\Documents\Studium\6. Semester\AML-Projekt\eigener Datensatz',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    subset='validation'
)

# CNN erstellen
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Modell kompilieren
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Modell trainieren
history = model.fit(train_generator,
                    epochs=10,
                    validation_data=validation_generator)

# Modell speichern
model.save('gesture_model_own_data.h5')