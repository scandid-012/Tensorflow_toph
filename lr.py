import tensorflow as tf
import numpy as np

# A basic sequential model with one Dense layer
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

# Compile the model with optimizer and loss
model.compile(optimizer='sgd', loss='mse')

# Training data
x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss',       # what to monitor (you can also use 'val_loss' if you have validation data)
    patience=20,          # number of epochs with no improvement before stopping
    restore_best_weights=True # restores weights from the best epoch
)

# Train the model with early stopping
history = model.fit(x, y, epochs=500, callbacks=[early_stop], verbose=1)

# Test prediction
print(model.predict(np.array([10.0])))
