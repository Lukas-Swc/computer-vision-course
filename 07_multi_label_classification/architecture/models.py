import tensorflow as tf


class VGGNetSmall():
    def __init__(self, input_shape, num_classes, final_activation):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.finale_activation = final_activation

    def build(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1024, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(units=self.num_classes, activation=self.finale_activation)
        ])

        return model