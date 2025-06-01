from flask import Flask, request, jsonify
# import joblib # No longer needed for PyTorch .pt models
import torch
import numpy as np # Still useful for array manipulations if needed before tensor conversion
import os
from PIL import Image # For handling image inputs
import io # For handling image bytes
from flask_cors import CORS # Import CORS
from ultralytics import YOLO # For loading YOLO models

# Initialize the Flask application
app = Flask(__name__)
CORS(app) # Enable CORS for all routes and origins by default

# --- Configuration ---
# It's good practice to load configurations from environment variables or a config file
# For simplicity, we define the model path directly here.
MODEL_FILENAME = 'best.pt' # Updated to your PyTorch model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

# --- Model Loading ---
# Load your pre-trained model once when the application starts.
model = None
TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """
    Loads the YOLO model from the specified path for a segmentation task.
    """
    global model
    print(f"* Attempting to load YOLO model for segmentation task on device: {TARGET_DEVICE}")
    try:
        if os.path.exists(MODEL_PATH):
            # Load the YOLO model using Ultralytics library, specifying the task and device
            model = YOLO(MODEL_PATH, task='segment')
            model.to(TARGET_DEVICE)  # Move model to the specified device (CPU or CUDA)
            # model.warmup(imgsz=(1, 3, 640, 640)) # Optional: warmup the model
            print(f"* YOLO segmentation model loaded successfully from {MODEL_PATH}")
            print(f"* Model is running on device: {model.device}")
            if str(model.device).split(':')[0] != TARGET_DEVICE.split(':')[0] and TARGET_DEVICE == "cuda":
                 print(f"* WARNING: Requested CUDA, but model is on {model.device}. Check CUDA setup.")
        else:
            print(f"* WARNING: Model file not found at {MODEL_PATH}. The /predict endpoint will not work.")
            print(f"* Please ensure '{MODEL_FILENAME}' is in the same directory as app.py or update MODEL_PATH.")
            model = None # Ensure model is None if file not found
    except Exception as e:
        print(f"* ERROR: Could not load YOLO segmentation model. Error: {e}")
        model = None # Ensure model is None if loading fails

# Call load_model when the application starts
load_model()

# --- (Placeholder) Preprocessing Function ---
def preprocess_input(image_bytes):
    """
    For YOLO, this function primarily loads the image.
    The YOLO model itself handles most transformations.

    Args:
        image_bytes (bytes): Raw bytes of the image file.

    Returns:
        A PIL Image object, or raises ValueError if image cannot be opened.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # You might want to convert to RGB if your model expects it,
        # though many YOLO models handle various formats.
        # if image.mode != 'RGB':
        #     image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Could not open or read image file: {str(e)}")

# --- (Placeholder) Postprocessing Function ---
def postprocess_output(results):
    """
    Postprocesses the YOLO model's segmentation results into a JSON-serializable format.

    Args:
        results: The output from the YOLO model's prediction (typically a Results object from Ultralytics).

    Returns:
        A dictionary containing a list of segmentations.
    """
    output_segmentations = []
    if results and len(results) > 0:
        # Assuming results is a list of Results objects (one per image, we send one image)
        result = results[0] # Get results for the first (and only) image
        names = result.names  # Class names

        if result.masks is not None: # Check if masks are present
            for i, mask in enumerate(result.masks.xy): # Iterate over polygon segments for each mask
                # Each 'mask' is a NumPy array of [N, 2] polygon points
                
                # Try to get associated class and confidence from boxes if available
                # This assumes that the order of masks corresponds to the order of boxes
                class_id = None
                confidence = None
                class_name = "unknown"

                if result.boxes and i < len(result.boxes):
                    box = result.boxes[i]
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = names[class_id]

                segment_info = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "polygon_points_xy": mask.tolist() # List of [x, y] points
                }
                output_segmentations.append(segment_info)
    return {"segmentations": output_segmentations}

# --- API Endpoint for Predictions ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "YOLO Model not loaded. Please check server logs or ensure best.pt exists."}), 503

    if 'image' not in request.files:
        return jsonify({"error": "No image file found in the request. Please upload an image with key 'image'."}), 400

    image_file = request.files['image']

    # Basic check for allowed file types (optional but good practice)
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    filename = image_file.filename
    if not ('.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({"error": "Invalid image file type. Allowed types: png, jpg, jpeg"}), 400

    try:
        image_bytes = image_file.read()

        # Preprocess the input image (primarily load it)
        # The YOLO model itself will handle resizing, normalization, etc.
        pil_image = preprocess_input(image_bytes)
        print(f"* Received image for prediction: {filename}, size: {pil_image.size}")

        # Make prediction with YOLO model
        # The model can take various inputs: PIL image, file path, numpy array, torch tensor
        # For multiple images, pass a list: model([pil_image1, pil_image2])
        results = model(pil_image, verbose=False) # verbose=False to reduce console output from YOLO
        
        print(f"* Model raw prediction results obtained.")

        # Postprocess the prediction
        processed_output = postprocess_output(results)
        # print(processed_output)

        return jsonify(processed_output)

    except ValueError as ve: # Catch specific preprocessing/validation errors
        print(f"* Value Error during prediction: {ve}")
        return jsonify({"error": "Invalid input data or image.", "details": str(ve)}), 400
    except Exception as e:
        print(f"* Unhandled error during prediction: {e}")
        import traceback
        traceback.print_exc() # For detailed debugging in logs
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)}), 500

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200

if __name__ == '__main__':
    # For development: host='0.0.0.0' makes it accessible on your network
    # For production, use a proper WSGI server like Gunicorn.
    app.run(host='0.0.0.0', port=5000, debug=True) # Set debug=False in production!