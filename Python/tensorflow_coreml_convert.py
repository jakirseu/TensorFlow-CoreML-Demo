# Install necessary libraries
!pip install tensorflow==2.10.0
!pip install coremltools==7.0b1

import numpy as np
import tensorflow as tf


# training data
celsius    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit = np.array([-40,  14, 32, 46.4, 59, 71.6, 100.4],  dtype=float)


# Build a simple model using TensorFlow (Sequential API)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')

# Train the model
history = model.fit(celsius, fahrenheit, epochs=50, verbose=0)

# Test the model on a value
print(model.predict(np.array([100])))

# Save the trained model
model.save('celsius_to_fahrenheit.h5')

import coremltools as ct
import tensorflow as tf

# Load the TensorFlow model (Keras Sequential model)
loaded_model = tf.keras.models.load_model('celsius_to_fahrenheit.h5')

# Convert the model to Core ML format using the correct input placeholder


coreml_model = ct.convert(
    loaded_model,
    source="tensorflow",
    # each time you run create the model, input name will change to someting like
    # dense_1_input, dense_2_input.. so you need to change the input name here.
    inputs=[ct.TensorType(shape=(1, 1), name="dense_4_input")]
)

# Save the Core ML model
coreml_model.save('CelsiusToFahrenheit.mlmodel')