import cv2
import numpy as np
import glob
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split


def create_CNNmodel():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 120, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


if __name__ == "__main__":
    dataset = 'data/thumbsup'
    dataset_path = os.path.join(dataset, '*')
    dataset_path = glob.glob(dataset_path)

    imageset = list()
    for path in dataset_path:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, [100, 120])
        imageset.append(gray)

    model = create_CNNmodel()
    output = [[1, 0]] * len(imageset)
    training_data = np.asarray(imageset)
    labels = np.asarray(output)
    X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.2, random_state=42)
    X_train = X_train.reshape(len(X_train), 100, 120, 1)
    X_test = X_test.reshape(len(X_test), 100, 120, 1)
    print(X_train.shape)
    print(y_train.shape)
    model.fit(X_train, y_train,
                batch_size=128,
                epochs=5,
                verbose=1,
                validation_data=(X_test, y_test))

    model.save("thumpsUP_model.h5")
