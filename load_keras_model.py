import tensorflow as tf

# Replace 'your_model.keras' with the path to your .keras file
model_path = "sentiment_model_en.keras"

# Load the Keras model
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    
    # Display the model summary
    model.summary()

except Exception as e:
    print(f"Error loading the model: {e}")
