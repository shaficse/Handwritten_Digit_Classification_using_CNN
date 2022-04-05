import tensorflow as tf

learning_rate = 1e-3
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
BATCH_SIZE = 64
EPOCHS = 100