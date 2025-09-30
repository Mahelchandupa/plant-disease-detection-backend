from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import io
import base64
from PIL import Image
import os
from werkzeug.utils import secure_filename
import logging
import gdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Class names for predictions
CLASS_NAMES = ['Healthy', 'Early Blight', 'Late Blight', 'Bacterial Spot']

# Global variables for models
cnn_model = None
transfer_model = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_transfer_architecture():
    """Create the exact transfer learning architecture - matching training setup"""
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',  # CRITICAL: Must match training
        include_top=False,
        input_shape=(128, 128, 3)
    )
    
    # Freeze the same layers as during training
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    
    # Don't compile yet - we'll do that after loading weights
    return model

def load_transfer_model_robust():
    """Load transfer model from numpy weights"""
    
    weights_path = 'models/transfer_weights_verified.npy'
    
    if os.path.exists(weights_path):
        try:
            logger.info("Loading verified model weights...")
            
            # Load the export
            export_data = np.load(weights_path, allow_pickle=True).item()
            
            # Check which format we have
            if 'trained_weights' in export_data:
                trained_weights = export_data['trained_weights']
            elif 'weights' in export_data:
                trained_weights = export_data['weights']
            else:
                raise KeyError("No weights found in file")
            
            # Create architecture
            model = create_transfer_architecture()
            model.build((None, 128, 128, 3))
            
            # Set the trained weights directly
            model.set_weights(trained_weights)
            
            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("✓ Transfer model loaded from verified weights")
            
            return model
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    else:
        logger.error(f"Weights file not found")
        return None
    
def verify_model_trained(model):
    """Verify the model has trained weights"""
    try:
        test_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
        test_pred = model.predict(test_input, verbose=0)
        max_prob = np.max(test_pred)
        
        if max_prob > 0.95:
            logger.warning(f"⚠️ Model shows high confidence ({max_prob:.3f}) on random input - may be untrained")
        else:
            logger.info(f"✓ Model verification passed (random input confidence: {max_prob:.3f})")
    except Exception as e:
        logger.warning(f"Model verification failed: {e}")
    
    return model

def load_models():
    """Load both trained models directly from local files"""
    global cnn_model, transfer_model

    try:
        # CNN model
        cnn_path = 'models/working_cnn_final.keras'
        if os.path.exists(cnn_path):
            cnn_model = tf.keras.models.load_model(cnn_path)
            logger.info("✅ CNN model loaded successfully")
        else:
            logger.error("❌ CNN model file not found")

        # Transfer model
        transfer_path = 'models/transfer_weights_verified.npy'
        if os.path.exists(transfer_path):
            transfer_model = load_transfer_model_robust()
            logger.info("✅ Transfer model loaded successfully")
        else:
            logger.error("❌ Transfer weights file not found")

    except Exception as e:
        logger.error(f"💥 Critical error in model loading: {str(e)}")
        cnn_model, transfer_model = None, None

def preprocess_image(image_data):
    """Preprocess image for model prediction"""
    try:
        # Convert bytes to PIL Image
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize to model input size (128x128)
        img_resized = cv2.resize(img_array, (128, 128))
        
        # Normalize to [0,1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_disease(image_batch, model_type='both'):
    """Make predictions using specified model(s)"""
    results = {}
    
    try:
        if model_type in ['cnn', 'both'] and cnn_model is not None:
            # CNN predictions
            cnn_predictions = cnn_model.predict(image_batch, verbose=0)
            cnn_probabilities = cnn_predictions[0]
            cnn_predicted_class = np.argmax(cnn_probabilities)
            cnn_confidence = float(np.max(cnn_probabilities))
            
            results['cnn'] = {
                'predicted_class': CLASS_NAMES[cnn_predicted_class],
                'confidence': cnn_confidence,
                'all_probabilities': {
                    CLASS_NAMES[i]: float(cnn_probabilities[i]) 
                    for i in range(len(CLASS_NAMES))
                }
            }
        
        if model_type in ['transfer', 'both'] and transfer_model is not None:
            # Transfer Learning predictions
            transfer_predictions = transfer_model.predict(image_batch, verbose=0)
            transfer_probabilities = transfer_predictions[0]
            transfer_predicted_class = np.argmax(transfer_probabilities)
            transfer_confidence = float(np.max(transfer_probabilities))
            
            results['transfer'] = {
                'predicted_class': CLASS_NAMES[transfer_predicted_class],
                'confidence': transfer_confidence,
                'all_probabilities': {
                    CLASS_NAMES[i]: float(transfer_probabilities[i]) 
                    for i in range(len(CLASS_NAMES))
                }
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = {
        'cnn_loaded': cnn_model is not None,
        'transfer_loaded': transfer_model is not None
    }
    return jsonify({
        'status': 'healthy',
        'models': model_status
    })

@app.route('/predict/cnn', methods=['POST'])
def predict_cnn():
    """Endpoint for CNN model predictions"""
    return predict_endpoint('cnn')

@app.route('/predict/transfer', methods=['POST'])
def predict_transfer():
    """Endpoint for Transfer Learning model predictions"""
    return predict_endpoint('transfer')

@app.route('/predict/both', methods=['POST'])
def predict_both():
    """Endpoint for both model predictions"""
    return predict_endpoint('both')

def predict_endpoint(model_type):
    """Generic prediction endpoint"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG files.'}), 400
        
        # Check model availability
        if model_type == 'cnn' and cnn_model is None:
            return jsonify({'error': 'CNN model not available'}), 503
        elif model_type == 'transfer' and transfer_model is None:
            return jsonify({'error': 'Transfer Learning model not available'}), 503
        elif model_type == 'both' and cnn_model is None and transfer_model is None:
            return jsonify({'error': 'No models available'}), 503
        
        # Process image
        image_data = file.read()
        processed_image = preprocess_image(image_data)
        
        # Make predictions
        predictions = predict_disease(processed_image, model_type)
        
        # Convert image to base64 for frontend display
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        response = {
            'success': True,
            'predictions': predictions,
            'image_data': f"data:image/jpeg;base64,{image_base64}",
            'filename': secure_filename(file.filename)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/model-info')
def model_info():
    """Get information about loaded models"""
    info = {
        'models': {
            'cnn': {
                'loaded': cnn_model is not None,
                'name': 'Custom CNN',
                'size': '6.1 MB',
                'parameters': '526,020'
            },
            'transfer': {
                'loaded': transfer_model is not None,
                'name': 'VGG16 Transfer Learning',
                'size': '114.8 MB',
                'parameters': '15,113,796'
            }
        },
        'classes': CLASS_NAMES,
        'input_size': '128x128x3',
        'accuracy': {
            'cnn': '45.6%',
            'transfer': '97.5%'
        }
    }
    
    return jsonify(info)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load models on startup
    load_models()
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get("PORT", 5000))
    
    # Run the application
    app.run(debug=False, host='0.0.0.0', port=port)