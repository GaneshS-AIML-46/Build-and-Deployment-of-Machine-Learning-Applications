"""
Script to fix model compatibility issues by converting the model to a newer format.
This addresses the 'batch_shape' parameter issue in InputLayer.
"""

import tensorflow as tf
import numpy as np

def fix_model_compatibility(input_model_path, output_model_path):
    """
    Load a model with compatibility issues and save it in a compatible format.
    """
    try:
        print(f"Attempting to load model from: {input_model_path}")
        
        # Try different loading approaches
        approaches = [
            lambda: tf.keras.models.load_model(input_model_path, compile=False),
            lambda: tf.keras.models.load_model(input_model_path, custom_objects={}, compile=False),
        ]
        
        model = None
        for i, approach in enumerate(approaches):
            try:
                print(f"Trying approach {i+1}...")
                model = approach()
                print(f"Success with approach {i+1}!")
                break
            except Exception as e:
                print(f"Approach {i+1} failed: {e}")
                continue
        
        if model is None:
            print("All loading approaches failed. Manual reconstruction needed.")
            return False
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        
        # Save in the new format
        print(f"Saving fixed model to: {output_model_path}")
        model.save(output_model_path)
        
        # Test the fixed model
        print("Testing the fixed model...")
        test_model = tf.keras.models.load_model(output_model_path)
        print("Fixed model loads successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error fixing model: {e}")
        return False

if __name__ == "__main__":
    input_path = "pneumonia_model.h5"
    output_path = "pneumonia_model_fixed.h5"
    
    print("Model Compatibility Fixer")
    print("=" * 50)
    
    success = fix_model_compatibility(input_path, output_path)
    
    if success:
        print(f"\n✅ Model successfully fixed and saved as '{output_path}'")
        print("You can now update your Flask app to use the fixed model.")
    else:
        print(f"\n❌ Could not fix the model automatically.")
        print("The model may need to be retrained with a compatible TensorFlow version.")
        print("For now, the demo model will work for testing the Flask app.")