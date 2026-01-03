
# House Prices Prediction Project

## Overview
This project predicts house prices using machine learning. It includes a complete workflow from data analysis to API deployment with containerization.

## Project Structure
```
├── house_prices.ipynb      # ML model development & training
├── app.py                  # FastAPI application with authentication
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
└── figures/                # Project visualizations
```

## Workflow

### 1. Machine Learning Pipeline (`house_prices.ipynb`)
- **Data Exploration**: Load and analyze housing dataset
- **Feature Engineering**: Process and transform features
- **Model Training**: Build and train prediction models
- **Evaluation**: Assess model performance

![Model Architecture](./figures/model_architecture.png)
![Training Results](./figures/training_results.png)

### 2. API Development (`app.py`)
FastAPI application with JWT authentication:

```python
from fastapi import FastAPI, Depends
from fastapi.security import HTTPBearer

app = FastAPI()
security = HTTPBearer()

@app.post("/predict")
async def predict(features: dict, token = Depends(security)):
    # Model prediction logic
    return {"predicted_price": price}
```

**Endpoints:**
- `POST /login` - User authentication
- `POST /predict` - Get price predictions
- `GET /health` - Health check

### 3. Containerization (`Dockerfile`)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

## Deployment
```bash
docker build -t house-prices:latest .
docker run -p 8000:8000 house-prices:latest
```

Access API at `http://localhost:8000/docs`
