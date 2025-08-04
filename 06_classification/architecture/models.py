import tensorflow as tf


class LeNet5:

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=120, activation='relu'),
            tf.keras.layers.Dense(units=84, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])

        return model
