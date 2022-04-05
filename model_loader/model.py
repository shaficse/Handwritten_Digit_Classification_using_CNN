import tensorflow as tf

def build_cnn_model():
    cnn_model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(filters=24, kernel_size=(3,3), activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
            tf.keras.layers.Conv2D(filters=36, kernel_size=(3,3), activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ]
    )
    return cnn_model

