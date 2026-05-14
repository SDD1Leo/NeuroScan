import os
import logging
import gdown

from tensorflow import keras

logger = logging.getLogger(__name__)

# Google Drive direct link - Replace with your actual model URL
# To get direct link: Share file -> Anyone with link -> Copy link -> replace 'view' with 'uc'
MODEL_URL = "https://drive.google.com/uc?id=1fi4r4Af0gxityCKGfIN-_dD7CQoyO-Aj"

# Model file path using .keras format (TensorFlow's recommended format)
# .keras is the modern successor to .h5 - it uses ZipFile-based serialization
# which is more reliable, supports more features, and has better compatibility
MODEL_PATH = "models/brain_tumor_model.keras"

# Global model variable - will be loaded on startup
model = None


def setup_tensorflow_for_low_memory():
    """
    Optimize TensorFlow for low-memory environments (like Render free tier).
    This reduces memory footprint significantly.
    """
    # Disable GPU - use CPU only (more reliable on free tiers)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Suppress TensorFlow info/warning messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Reduce memory growth - don't pre-allocate all memory
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

    # Disable XLA compilation for faster startup on small instances
    os.environ["TF_ENABLE_XLA"] = "0"


def create_models_directory():
    """
    Create the models directory if it doesn't exist.
    This ensures we have a place to download the model file.
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created models directory at: {models_dir}")
    return models_dir


def download_model_from_google_drive():
    """
    Download the .keras model from Google Drive using gdown.

    Download Flow:
    1. Check if model already exists locally (skip download if exists)
    2. Use gdown to download large files from Google Drive
    3. gdown handles large file downloads better than direct HTTP
    4. Downloaded .keras file is saved to models/ directory

    Why .keras format?
    - Modern TensorFlow native format (TensorFlow 2.x)
    - Uses ZipFile-based serialization (more reliable than HDF5)
    - Better support for custom layers, metrics, and model components
    - Recommended by TensorFlow team over legacy .h5 format

    Note: For Google Drive links, ensure you use the direct download link:
    https://drive.google.com/uc?id=FILE_ID (not the view URL)
    """
    if os.path.exists(MODEL_PATH):
        logger.info(f"Model already exists at {MODEL_PATH}, skipping download.")
        return True

    if MODEL_URL == "PASTE_GOOGLE_DRIVE_DIRECT_LINK_HERE":
        logger.warning("Model URL not configured! Please set MODEL_URL in model_loader.py")
        return False

    logger.info(f"Downloading model from Google Drive: {MODEL_URL}")

    try:
        # gdown downloads files from Google Drive
        # -O specifies output file path
        # -f forces overwrite if exists
        gdown.download(MODEL_URL, MODEL_PATH, fuzzy=True)
        logger.info(f"Model downloaded successfully to {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        return False


def load_model_into_memory():
    """
    Load the Keras model into memory for inference.

    Model Loading Flow:
    1. First ensure TensorFlow is optimized for low memory/CPU-only
    2. Import load_model from tensorflow.keras.models
    3. Load .keras model file using load_model with compile=False
    4. Model is loaded globally and reused for all predictions
    5. This avoids reloading on every request (performance optimization)

    Why compile=False?
    - Faster loading (skips optimizer initialization)
    - Lower memory footprint (no optimizer state stored)
    - Suitable for inference-only scenarios like this API
    - We don't need to retrain, just predict

    Returns:
        Loaded Keras model or None if loading fails
    """
    global model

    # Optimize TensorFlow first for CPU-only, low-memory operation
    # This must be done BEFORE importing Keras to take effect
    setup_tensorflow_for_low_memory()

    # Import here to ensure environment variables are set first
    from tensorflow.keras.models import load_model

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        return None

    try:
        logger.info(f"Loading Keras model from {MODEL_PATH}")
        # Load .keras model with compile=False for faster, memory-efficient loading
        # compile=False skips optimizer initialization - ideal for inference
        model = load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None


def get_model():
    """
    Get the globally loaded model instance.

    Returns:
        The loaded Keras model, or None if not loaded
    """
    return model


def initialize_model():
    """
    Main initialization function - orchestrates the entire model loading process.

    This is called on application startup to:
    1. Create models directory
    2. Download model from Google Drive (if not exists)
    3. Load model into memory

    Returns:
        Loaded model or None
    """
    logger.info("Initializing model loader...")

    # Step 1: Create models directory
    create_models_directory()

    # Step 2: Download model if needed
    download_success = download_model_from_google_drive()
    if not download_success:
        logger.error("Model download failed. Please check MODEL_URL.")
        return None

    # Step 3: Load model into memory
    loaded_model = load_model_into_memory()
    if loaded_model is None:
        logger.error("Model loading failed.")
        return None

    logger.info("Model initialization complete!")
    return loaded_model