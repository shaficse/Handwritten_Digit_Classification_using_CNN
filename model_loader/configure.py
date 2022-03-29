import tensorflow as tf

learning_rate = 1e-1
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']
optimizer = tf.keras.optimizers.SGD(learning_rate)
BATCH_SIZE = 64
EPOCHS = 5