# NeuroScan - Brain Tumor Detection API

Production-ready FastAPI backend for brain tumor detection using deep learning.

## Features

- FastAPI-based REST API
- Automatic model download from Google Drive on startup
- Image preprocessing pipeline (224x224, normalized)
- 4-class classification: Glioma, Meningioma, No Tumor, Pituitary
- Optimized for low-memory environments (Render free tier compatible)
- Health check endpoints

## Project Structure

```
backend/
├── app.py              # FastAPI application and routes
├── model_loader.py    # Model download and loading logic
├── requirements.txt   # Python dependencies
├── runtime.txt        # Python version specification
├── render.yaml        # Render deployment configuration
├── .gitignore         # Git ignore rules
├── README.md          # This file
├── models/            # Model files (auto-created, not committed)
└── utils/             # Utility modules (reserved for future use)
```

## Local Setup

### Prerequisites

- Python 3.11.9
- pip (Python package manager)

### Installation

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure the model URL:
   - Open `model_loader.py`
   - Replace `MODEL_URL = "PASTE_GOOGLE_DRIVE_DIRECT_LINK_HERE"` with your Google Drive direct link

### Running Locally

Start the development server:
```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:10000`

## API Endpoints

### Root Endpoint (Health Check)
```bash
GET /
```

Response:
```json
{
  "status": "healthy",
  "message": "NeuroScan Brain Tumor Detection API is running",
  "version": "1.0.0",
  "endpoints": {
    "health": "/",
    "predict": "/predict (POST)"
  }
}
```

### Prediction Endpoint
```bash
POST /predict
Content-Type: multipart/form-data
```

Request: Upload an image file (JPEG, PNG, etc.)

Response:
```json
{
  "prediction": "Glioma",
  "confidence": 0.97
}
```

### Detailed Health Check
```bash
GET /health
```

## Example curl Request

```bash
curl -X POST "http://localhost:10000/predict" \
  -F "file=@/path/to/mri_image.jpg"
```

## Example Frontend Fetch Request

```javascript
const formData = new FormData();
formData.append('file', imageFile);

fetch('https://your-render-app.onrender.com/predict', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => {
    console.log('Prediction:', data.prediction);
    console.log('Confidence:', data.confidence);
  })
  .catch(error => console.error('Error:', error));
```

## Getting Google Drive Direct Link

To get a direct download link for your `.h5` model:

1. **Upload your model to Google Drive**

2. **Share the file**:
   - Right-click on the file → Share
   - Change "General access" to "Anyone with the link"

3. **Get the direct link**:
   - Copy the share link (looks like: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`)
   - Extract the FILE_ID (the part between `/d/` and `/view`)
   - Construct direct link: `https://drive.google.com/uc?id=FILE_ID`

4. **Example**:
   - Share link: `https://drive.google.com/file/d/1abc123def456Ghi789/view?usp=sharing`
   - Direct link: `https://drive.google.com/uc?id=1abc123def456Ghi789`

5. **Update model_loader.py**:
   ```python
   MODEL_URL = "https://drive.google.com/uc?id=1abc123def456Ghi789"
   ```

## Render Deployment

### Prerequisites

- GitHub, GitLab, or Bitbucket account
- Render account (free tier works)

### Deployment Steps

1. **Push to Git**:
   ```bash
   git add .
   git commit -m "Add NeuroScan backend"
   git push origin main
   ```

2. **Create Render Service**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Web Service"
   - Connect your Git repository

3. **Configure Service**:
   - Name: `neuroscan-backend`
   - Runtime: Python 3.11.9
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port 10000`

4. **Environment Variables** (optional):
   - `PYTHONUNBUFFERED=1`
   - `TF_CPP_MIN_LOG_LEVEL=2`

5. **Deploy**:
   - Click "Create Web Service"
   - Wait for build and deployment (first deployment downloads model ~100MB)

6. **Test**:
   - Visit `https://your-service-name.onrender.com/`
   - Test prediction at `https://your-service-name.onrender.com/predict`

### Render Free Tier Notes

- Free tier has limits on compute time and memory
- Model downloads on first request and is cached
- First prediction may take longer (model loading)
- Service sleeps after 15 minutes of inactivity
- First request after sleep may take longer (cold start)

## Model Classes

The model classifies MRI images into 4 categories:

| Index | Class     | Description                    |
|-------|-----------|--------------------------------|
| 0     | Glioma    | Glioma tumor                  |
| 1     | Meningioma| Meningioma tumor              |
| 2     | No Tumor  | Healthy (no tumor detected)   |
| 3     | Pituitary | Pituitary tumor               |

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Successful prediction
- `400` - Invalid file type or malformed request
- `500` - Server error (model loading, prediction failure)

## Performance Optimization

This backend includes several optimizations for low-memory environments:

- CPU-only inference (no GPU memory usage)
- Environment variable configuration to limit TensorFlow memory
- Model loaded once at startup and reused for all predictions
- Efficient image preprocessing with NumPy

## Troubleshooting

### Model Download Fails

- Check Google Drive link is correct and publicly accessible
- Verify file ID is correct in the direct link
- Check network connectivity

### Out of Memory on Render

- Free tier has ~512MB memory limit
- TensorFlow is configured for CPU-only inference
- Consider using a smaller model or optimizing

### Cold Start Issues

- First request after deployment takes longer
- Model downloads (~100MB) and loads on startup
- Subsequent requests are faster

## License

MIT License

## Support

For issues or questions, please open a GitHub issue.