import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    '''
    # Normalizacja danych
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Tworzenie modelu
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # poprawiono z (25,25) na (28,28)
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
   
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)
    '''
    #model.save('handwritten.h5')
    model = tf.keras.models.load_model('handwritten.h5')
    loss, accuracy = model.evaluate(x_test, y_test)

    print(loss,'\n', accuracy)

if __name__ == '__main__':
    main()
