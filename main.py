import numpy as np
import tensorflow as tf

def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizacja (0-1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Jeśli chcesz wytrenować model, odkomentuj poniższy blok:
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=5)

    model.save('handwritten.keras')
    """

    # Wczytaj wytrenowany model
    model = tf.keras.models.load_model('handwritten.keras')

    # Ewaluacja
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
