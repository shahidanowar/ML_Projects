import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the MNIST dataset (handwritten digits)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalize and reshape the data
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 3. Build a simple CNN model
model = tf.keras.Sequential([  # Create a sequential model (stack of layers)
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  
    # First convolutional layer with 32 filters, 3x3 kernel, ReLU activation, input size 28x28 with 1 channel (grayscale)

    tf.keras.layers.MaxPooling2D((2,2)),  
    # Max pooling layer to reduce spatial dimensions (2x2 pooling)

    tf.keras.layers.Flatten(),  
    # Flatten the 2D feature maps to 1D vector for the dense layers

    tf.keras.layers.Dense(64, activation='relu'),  
    # Fully connected (dense) layer with 64 neurons and ReLU activation

    tf.keras.layers.Dense(10, activation='softmax')  
    # Output layer with 10 neurons (for 10 digit classes) and softmax activation for probability distribution
])

# 4. Compile the model
model.compile(optimizer='adam',  
              # Use Adam optimizer (efficient and adaptive optimizer)

              loss='sparse_categorical_crossentropy',  
              # Use sparse categorical crossentropy for integer-labeled classification

              metrics=['accuracy'])  
              # Track accuracy as a performance metric during training


# 5. Train the model
model.fit(x_train, y_train, epochs=5)

# 6. Evaluate on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# 7. Recognize a digit from the test set
index = np.random.randint(0, len(x_test))
sample_image = x_test[index]
true_label = y_test[index]

prediction = model.predict(sample_image.reshape(1, 28, 28, 1))
predicted_label = np.argmax(prediction)

# 8. Show the result
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title(f"True: {true_label}, Predicted: {predicted_label}")
plt.axis('off')
plt.show()

model.save('digit_model.keras')
