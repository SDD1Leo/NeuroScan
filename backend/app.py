import logging
import io
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

# Import model loader
import model_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="NeuroScan Brain Tumor Detection API",
    description="AI-powered brain tumor detection using deep learning",
    version="1.0.0"
)

# Class names for the 4-class brain tumor classification
# These correspond to the output classes of the trained model
CLASS_NAMES = [
    "Glioma",      # Class 0
    "Meningioma",  # Class 1
    "No Tumor",    # Class 2
    "Pituitary"    # Class 3
]

# Image preprocessing constants
IMAGE_SIZE = (224, 224)  # Standard size for most CNN architectures (e.g., VGG, ResNet)


@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    Loads the ML model on server startup.

    Startup Flow:
    1. Log startup message
    2. Initialize and load the model
    3. Log success/failure status

    This ensures the model is ready for inference when first request arrives.
    """
    logger.info("=" * 50)
    logger.info("NeuroScan API Starting Up...")
    logger.info("=" * 50)

    # Initialize model - downloads from Google Drive if needed, then loads
    model = model_loader.initialize_model()

    if model is not None:
        logger.info("Model loaded successfully and ready for predictions!")
    else:
        logger.error("Failed to load model. API will return errors for prediction requests.")

    logger.info("=" * 50)
    logger.info("Startup complete!")
    logger.info("=" * 50)


@app.get("/")
async def root():
    """
    Root endpoint - Health check.

    Returns:
        JSON response with API status and information

    This endpoint is used to:
    - Verify the API is running
    - Check health status
    - Provide basic API information
    """
    return {
        "status": "healthy",
        "message": "NeuroScan Brain Tumor Detection API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/",
            "predict": "/predict (POST)"
        }
    }


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for model inference.

    Image Preprocessing Pipeline:
    1. Convert to RGB - ensures consistent 3-channel input
    2. Resize to 224x224 - matches model's expected input size
    3. Convert to NumPy array - required format for TensorFlow
    4. Normalize to [0, 1] - divide by 255 to scale pixel values
    5. Expand dimensions - add batch dimension (shape: 1, 224, 224, 3)

    Args:
        image: PIL Image object

    Returns:
        Preprocessed image as NumPy array with shape (1, 224, 224, 3)
    """
    # Step 1: Convert to RGB (handles grayscale, RGBA, palette modes)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Step 2: Resize to model input size (224x224)
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    # Step 3: Convert to NumPy array with float32 type
    image_array = np.array(image, dtype=np.float32)

    # Step 4: Normalize pixel values to [0, 1] range
    # This matches how the model was trained
    image_array = image_array / 255.0

    # Step 5: Expand dimensions to create batch dimension
    # Model expects shape: (batch_size, height, width, channels)
    # Adding batch dimension: (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def make_prediction(image_array: np.ndarray) -> dict:
    """
    Run inference on preprocessed image.

    Prediction Flow:
    1. Get the global model instance
    2. Run model.predict() on the preprocessed image
    3. Get prediction probabilities for each class
    4. Find the class with highest probability (argmax)
    5. Return label and confidence score

    Args:
        image_array: Preprocessed image as NumPy array

    Returns:
        Dictionary with prediction label and confidence
    """
    # Get the loaded model
    model = model_loader.get_model()

    if model is None:
        raise RuntimeError("Model not loaded. Cannot make prediction.")

    # Run inference
    # Returns array of shape (1, num_classes) with probabilities
    prediction_probabilities = model.predict(image_array, verbose=0)

    # Get the class with highest probability
    predicted_class_index = np.argmax(prediction_probabilities[0])
    confidence_score = float(prediction_probabilities[0][predicted_class_index])

    # Map index to class name
    predicted_label = CLASS_NAMES[predicted_class_index]

    return {
        "prediction": predicted_label,
        "confidence": round(confidence_score, 2)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict brain tumor class from uploaded MRI image.

    API Handling Flow:
    1. Receive image file via multipart/form-data
    2. Validate file is an image (check content type/extension)
    3. Read image bytes into memory
    4. Open image using Pillow
    5. Preprocess image (resize, normalize, expand dims)
    6. Run model inference
    7. Return prediction result as JSON

    Args:
        file: Uploaded image file (multipart/form-data)

    Returns:
        JSON response with prediction and confidence

    Raises:
        HTTPException: If image is invalid or processing fails

    Response Format:
        {
            "prediction": "Glioma",  // Predicted class name
            "confidence": 0.97       // Confidence score (0-1)
        }
    """
    # Validate file is an image
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )

    try:
        # Read the uploaded file bytes
        contents = await file.read()

        # Open image using Pillow
        # This handles various image formats (JPEG, PNG, etc.)
        image = Image.open(io.BytesIO(contents))

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        result = make_prediction(preprocessed_image)

        logger.info(f"Prediction: {result['prediction']}, Confidence: {result['confidence']}")

        return JSONResponse(content=result)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Handle any other errors during prediction
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    Detailed health check endpoint.

    Returns:
        JSON response with health status and model loading state
    """
    model_loaded = model_loader.get_model() is not None

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": model_loader.MODEL_PATH
    }


if __name__ == "__main__":
    """
    Local development entry point.
    Run with: uvicorn app:app --reload

    Render Deployment Note:
    On Render, use the start command specified in render.yaml:
    uvicorn app:app --host 0.0.0.0 --port 10000

    The --host 0.0.0.0 ensures the server is accessible externally.
    The --port 10000 is required by Render's platform.
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)