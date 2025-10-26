import tensorflow as tf
import numpy as np

def create_realistic_demo_model():
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    
    # Convert to grayscale for medical image analysis
    grayscale = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(inputs)
    
    # Feature extraction layers
    x = tf.keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same')(grayscale)
    x = tf.keras.layers.MaxPooling2D((4, 4))(x)
    
    x = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((4, 4))(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_synthetic_training_data():
    """Create synthetic training data to train the demo model"""
    # Create synthetic X-ray-like images
    X_train = []
    y_train = []
    
    np.random.seed(42)  # For reproducible results
    
    # Generate 1000 synthetic images
    for i in range(1000):
        # Create base image with lung-like structure
        img = np.random.normal(0.3, 0.1, (224, 224, 3))  # Darker base
        
        # Add some structure (simulating ribs, lung boundaries)
        for j in range(5):
            y_pos = int(np.random.uniform(50, 174))
            img[y_pos:y_pos+2, :, :] += 0.2  # Horizontal lines (ribs)
        
        # Decide if this is pneumonia or normal
        is_pneumonia = i % 2 == 0  # 50% pneumonia, 50% normal
        
        if is_pneumonia:
            # Add cloudy/opaque regions (pneumonia indicators)
            num_spots = np.random.randint(3, 8)
            for _ in range(num_spots):
                x_center = np.random.randint(50, 174)
                y_center = np.random.randint(50, 174)
                radius = np.random.randint(15, 40)
                
                # Create circular cloudy region
                y, x = np.ogrid[:224, :224]
                mask = (x - x_center)**2 + (y - y_center)**2 <= radius**2
                img[mask] += np.random.uniform(0.2, 0.4)
        else:
            # Normal lung - clearer regions
            img += np.random.normal(0, 0.05, (224, 224, 3))
        
        # Normalize to [0, 1]
        img = np.clip(img, 0, 1)
        
        X_train.append(img)
        y_train.append(1 if is_pneumonia else 0)
    
    return np.array(X_train), np.array(y_train)

if __name__ == "__main__":
    print("Creating realistic demo pneumonia detection model...")
    
    # Create model
    model = create_realistic_demo_model()
    print("Model architecture created.")
    
    # Create synthetic training data
    print("Generating synthetic training data...")
    X_train, y_train = create_synthetic_training_data()
    print(f"Created {len(X_train)} training samples")
    
    # Train the model briefly
    print("Training model on synthetic data...")
    model.fit(X_train, y_train, 
              epochs=10, 
              batch_size=32, 
              validation_split=0.2,
              verbose=1)
    
    # Save the trained model
    model.save('realistic_demo_pneumonia_model.h5')
    print("Realistic demo model saved as 'realistic_demo_pneumonia_model.h5'")
    
    # Test the model
    print("\nTesting model predictions...")
    test_predictions = model.predict(X_train[:10])
    for i, (pred, actual) in enumerate(zip(test_predictions, y_train[:10])):
        pred_class = "Pneumonia" if pred[0] > 0.5 else "Normal"
        actual_class = "Pneumonia" if actual == 1 else "Normal"
        confidence = pred[0] if pred[0] > 0.5 else 1 - pred[0]
        print(f"Sample {i+1}: Predicted: {pred_class} ({confidence:.1%}), Actual: {actual_class}")
    
    print("\n✅ Realistic demo model created and trained successfully!")