from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Try to load the model with different approaches
model = None
model_error = None
model_name = "unknown"

# Try original model first, then fallback to demo models
model_files = ['pneumonia_model.h5', 'realistic_demo_pneumonia_model.h5', 'demo_pneumonia_model.h5']

for model_file in model_files:
    try:
        import tensorflow as tf
        # Try loading with compile=False to avoid optimizer issues
        model = tf.keras.models.load_model(model_file, compile=False)
        model_name = model_file
        print(f"Model '{model_file}' loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        break
    except Exception as e:
        print(f"Error loading model '{model_file}': {e}")
        model_error = str(e)
        continue

if model is None:
    print("Failed to load any model. The app will run but predictions won't work.")
    print("Consider running 'python create_demo_model.py' to create a working demo model.")

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size (assuming 224x224, adjust if different)
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model-status')
def model_status():
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded.',
            'error_details': model_error if model_error else 'Unknown error'
        }), 500
    else:
        return jsonify({
            'status': 'success',
            'message': f'Model loaded successfully: {model_name}',
            'model_file': model_name,
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape)
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check your model file and dependencies.'}), 500
        
        # Get the uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Process the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Assuming binary classification: 0 = Normal, 1 = Pneumonia
        raw_confidence = float(prediction[0][0])
        
        if raw_confidence > 0.5:
            result = "Pneumonia Detected"
            risk_level = "High"
            confidence = raw_confidence
        else:
            result = "Normal"
            risk_level = "Low"
            confidence = 1 - raw_confidence
        
        # Add some additional context based on confidence levels
        if confidence > 0.9:
            certainty = "Very High"
        elif confidence > 0.8:
            certainty = "High"
        elif confidence > 0.7:
            certainty = "Moderate"
        else:
            certainty = "Low"
        
        return jsonify({
            'result': result,
            'confidence': f"{confidence * 100:.1f}%",
            'risk_level': risk_level,
            'certainty': certainty,
            'raw_score': f"{raw_confidence:.3f}",
            'model_used': model_name
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)