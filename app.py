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

def download_models():
    """Download models from Google Drive with detailed logging"""
    model_files = {
        'transfer_weights_verified.npy': '1xYQK1qhYFSS799b2GyrjFzTyyqH0q2uy',
        'working_cnn_final.keras': '1NAXMeUG8kp5NYbu5C-Pu_g_wdY2V7cIM'
    }
    
    os.makedirs('models', exist_ok=True)
    logger.info("📁 Models directory ready")
    
    for filename, file_id in model_files.items():
        filepath = f'models/{filename}'
        
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
            logger.info(f"✅ {filename} already exists ({file_size:.2f} MB)")
            continue
            
        if file_id.startswith('YOUR_ACTUAL'):
            logger.error(f"❌ {filename}: File ID not configured")
            continue
            
        try:
            logger.info(f"📥 Downloading {filename}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            
            # Download with progress
            gdown.download(url, filepath, quiet=False)
            
            # Verify download
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"✅ Successfully downloaded {filename} ({file_size:.2f} MB)")
            else:
                logger.error(f"❌ Download failed: {filename} not created")
                
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {str(e)}")

def download_models_with_fallback():
    """Try main download first, then fallback to alternative method"""
    os.makedirs('models', exist_ok=True)
    
    model_files = {
        'transfer_weights_verified.npy': '1xYQK1qhYFSS799b2GyrjFzTyyqH0q2uy',
        'working_cnn_final.keras': '1NAXMeUG8kp5NYbu5C-Pu_g_wdY2V7cIM'
    }
    
    # Track which files need downloading
    files_to_download = []
    for filename, file_id in model_files.items():
        filepath = f'models/{filename}'
        if not os.path.exists(filepath):
            files_to_download.append((filename, file_id))
            logger.info(f"📋 Need to download: {filename}")
        else:
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            logger.info(f"✅ Already exists: {filename} ({file_size:.2f} MB)")
    
    if not files_to_download:
        logger.info("🎉 All model files already downloaded")
        return True
    
    # Try main gdown method first
    logger.info("🔄 Attempting main download with gdown...")
    success_count = 0
    
    for filename, file_id in files_to_download:
        filepath = f'models/{filename}'
        try:
            logger.info(f"📥 Main download: {filename}")
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filepath, quiet=False)
            
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"✅ Main download success: {filename} ({file_size:.2f} MB)")
                success_count += 1
            else:
                logger.warning(f"⚠️ Main download failed: {filename}")
                
        except Exception as e:
            logger.warning(f"⚠️ Main download failed for {filename}: {e}")
    
    # Check if we need fallback
    remaining_files = [(f, fid) for f, fid in files_to_download if not os.path.exists(f'models/{f}')]
    
    if remaining_files:
        logger.info(f"🔄 {len(remaining_files)} files failed, trying alternative download...")
        success_count += download_models_alternative(remaining_files)
    
    total_files = len(files_to_download)
    logger.info(f"📊 Download summary: {success_count}/{total_files} files successful")
    
    return success_count == total_files

def download_models_alternative(specific_files=None):
    """Alternative download method using requests"""
    import requests
    
    model_files = {
        'transfer_weights_verified.npy': '1xYQK1qhYFSS799b2GyrjFzTyyqH0q2uy',
        'working_cnn_final.keras': '1NAXMeUG8kp5NYbu5C-Pu_g_wdY2V7cIM'
    }
    
    # Use specific files if provided, otherwise all files
    if specific_files:
        files_to_download = specific_files
    else:
        files_to_download = [(f, fid) for f, fid in model_files.items() if not os.path.exists(f'models/{f}')]
    
    success_count = 0
    
    for filename, file_id in files_to_download:
        filepath = f'models/{filename}'
        
        if os.path.exists(filepath):
            continue
            
        try:
            logger.info(f"🔄 Alternative download: {filename}")
            
            # Direct download with confirmation handling
            url = f"https://docs.google.com/uc?export=download&id={file_id}"
            session = requests.Session()
            response = session.get(url, stream=True)
            
            # Handle large file confirmation
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    url = f"https://docs.google.com/uc?export=download&confirm={value}&id={file_id}"
                    response = session.get(url, stream=True)
                    break
            
            # Save file with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Log progress for large files
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            if progress % 20 == 0:  # Log every 20%
                                logger.info(f"📦 Downloading {filename}: {progress:.1f}%")
            
            # Verify download
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"✅ Alternative download successful: {filename} ({file_size:.2f} MB)")
                success_count += 1
            else:
                logger.error(f"❌ Alternative download failed: {filename}")
                
        except Exception as e:
            logger.error(f"❌ Alternative download error for {filename}: {e}")
    
    return success_count

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
    """Load both trained models with better error handling"""
    global cnn_model, transfer_model
    
    try:
        # Download models first
        download_success = download_models_with_fallback()
        
        if not download_success:
            logger.error("💥 Model download failed - check file IDs and internet connection")
        
        # Check which files actually exist
        model_paths = {
            'CNN': 'models/working_cnn_final.keras',
            'Transfer': 'models/transfer_weights_verified.npy'
        }
        
        for model_name, path in model_paths.items():
            if os.path.exists(path):
                file_size = os.path.getsize(path) / (1024 * 1024)
                logger.info(f"📊 {model_name} model: {path} ({file_size:.2f} MB)")
            else:
                logger.warning(f"⚠️ {model_name} model not found: {path}")
        
        # Load CNN model
        cnn_path = 'models/working_cnn_final.keras'
        if os.path.exists(cnn_path):
            try:
                cnn_model = tf.keras.models.load_model(cnn_path)
                logger.info("✅ CNN model loaded successfully")
                
                # Test the model
                test_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
                test_pred = cnn_model.predict(test_input, verbose=0)
                logger.info(f"🧪 CNN test prediction shape: {test_pred.shape}")
                
            except Exception as e:
                logger.error(f"❌ CNN model loading failed: {e}")
                cnn_model = None
        else:
            logger.warning("⚠️ CNN model file not found")
            cnn_model = None
        
        # Load transfer model
        transfer_model = load_transfer_model_robust()
            
    except Exception as e:
        logger.error(f"💥 Critical error in model loading: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        cnn_model = None
        transfer_model = None

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