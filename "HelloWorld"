# a basic sequential model with simple inputs and only one layer of connected layer
model=tf.keras.Sequential([    
tf.keras.Input(shape=(1,)),
tf.keras.layers.Dense(Units=1)
])
# optizer and loss used to optimize the model
model.compile(optimizer='sgd', loss='mse')
x=np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y=np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
#fit is used to fit x to y(training the model)
model.fit(x, y, epochs=500)
# epochs 500 means the training loop will continue for 500 times
#testing the model using predict
model.predict(np.array([10]))

