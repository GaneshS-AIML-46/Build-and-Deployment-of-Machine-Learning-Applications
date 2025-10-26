import tensorflow as tf
import numpy as np

# Create a simple demo model for testing the Flask app
def create_demo_pneumonia_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    print("Creating demo pneumonia detection model...")
    model = create_demo_pneumonia_model()
    
    # Save the model
    model.save('demo_pneumonia_model.h5')
    print("Demo model saved as 'demo_pneumonia_model.h5'")
    
    # Test loading
    loaded_model = tf.keras.models.load_model('demo_pneumonia_model.h5')
    print("Demo model loaded successfully!")
    print(f"Input shape: {loaded_model.input_shape}")
    print(f"Output shape: {loaded_model.output_shape}")