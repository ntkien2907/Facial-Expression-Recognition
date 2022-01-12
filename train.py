import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from utils import save_figures

# Exact name of your folders
CLASSES = ['angry', 'happy', 'neutral', 'sad', 'surprise']
N_CLASSES = len(CLASSES)

IMG_SIZE = 128
BATCH_SIZE = 32
N_EPOCHS = 50
LEARNING_RATE = 1e-4
DECAY = 1e-6

# Preprocess data
def preprocess_data(dir, labels=CLASSES, img_size=IMG_SIZE):
    # Read all images in dir folder
    data, X, y = [], [], []
    for category in labels:
        path = os.path.join(dir, category)
        category_idx = labels.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                data.append([new_array, category_idx])
            except Exception as e:
                pass
    # Shuffle data
    np.random.shuffle(data)
    for features, label in data:
        X.append(features)
        y.append(label)
    # Convert all images and their labels to NumPy arrays
    X, y = np.array(X), np.array(y)
    # Normalize data and one-hot encode the corresponding labels
    X = X / 255.0
    y = to_categorical(y, num_classes=N_CLASSES)
    return X, y

# Training data
TRAIN_DIR = 'data/train/'
X_train, y_train = preprocess_data(dir=TRAIN_DIR)

# Testing data
TEST_DIR = 'data/test/'
X_test, y_test = preprocess_data(dir=TEST_DIR)

# Convolutional Neural Network
model = Sequential([
    # 1st layer
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    Dropout(0.25),

    # 2nd layer
    Conv2D(128, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    Dropout(0.25),

    # 3rd layer
    Conv2D(128, kernel_size=(3,3), activation='relu'),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    Dropout(0.25),

    # Fully connected layers
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),

    # Final layer
    Dense(N_CLASSES, activation='softmax')
])
model.summary()

# Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

# Compile model
opt = Adam(learning_rate=LEARNING_RATE, decay=DECAY)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

# Train model
history = model.fit(
    X_train, y_train, 
    batch_size=BATCH_SIZE, 
    steps_per_epoch=np.ceil(len(X_train)/BATCH_SIZE), 
    epochs=N_EPOCHS, 
    validation_data=(X_test, y_test), 
    validation_steps=np.ceil(len(X_test)/BATCH_SIZE), 
    callbacks=[early_stopping])

# Save history for accuracy and loss
save_figures(history, dir='figures/')

# Evaluate model
print('\n[INFO] evaluating network ...')
y_preds = model.predict(X_test, batch_size=BATCH_SIZE, steps=np.ceil(len(X_test)/BATCH_SIZE))
y_preds = np.argmax(y_preds, axis=1)
print(classification_report(y_test.argmax(axis=1), y_preds, target_names=CLASSES, zero_division=0))

# Save model
print('\n[INFO] saving model ...')
model.save('facial-expression-model.h5')