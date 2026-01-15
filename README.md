# Machine Learning Projects

End-to-end ML projects demonstrating classification, regression, and deep learning with production-ready API deployments.

---

## Projects

### 1. Iris Classification
**Status**: Complete | **Type**: Multi-class Classification

End-to-end ML pipeline for iris species classification with FastAPI deployment.

**Key Features**:
- Model comparison (SVM, Logistic Regression, Random Forest, Decision Tree)
- 96.7% accuracy with Support Vector Machine
- REST API with FastAPI + Docker

**Tech Stack**: scikit-learn, FastAPI, Docker

[View Project](./Iris-Project/)

---

### 2. House Prices Prediction
**Status**: Complete | **Type**: Regression

Kaggle house prices prediction with comprehensive feature engineering and production deployment.

**Key Features**:
- Advanced preprocessing (79 → 290 features after encoding)
- Ridge, Lasso, ElasticNet comparison with GridSearchCV
- Missing value imputation with domain knowledge
- Residual analysis and model diagnostics

**Tech Stack**: scikit-learn, FastAPI, Docker

[View Project](./HousePrices-Project/)

---

### 3. CNN Image Classifier (CIFAR-10)
**Status**: Complete | **Type**: Deep Learning / Image Classification

Multi-model comparison for image classification: Baseline CNN → Augmented CNN → Transfer Learning.

**Key Features**:
- **3-model progressive comparison** demonstrating iterative improvement
- **Transfer Learning with MobileNetV2** (87.4% accuracy)
- Data augmentation pipeline (flip, rotation, zoom, contrast)
- Batch Normalization for training stability
- **Overfitting analysis** and mitigation strategies
- REST API with single/batch prediction endpoints
- Docker containerization

**Tech Stack**: TensorFlow, Keras, MobileNetV2, FastAPI, Docker

[View Project](./CNN_image_classifier/)

---

## Repository Structure

```
MachineLearning/
├── Iris-Project/              # Classification with SVM
├── HousePrices-Project/       # Regression with Ridge/Lasso
├── CNN_image_classifier/      # Deep Learning with Transfer Learning
└── README.md
```

---

## Future Projects

- **Sentiment Analysis**: NLP with Transformers (IMDB reviews)
- **Time Series Forecasting**: Stock/weather prediction with LSTM
- **Recommender System**: Collaborative filtering

---

## Skills Demonstrated

**Machine Learning**: Supervised learning, model selection, cross-validation, hyperparameter tuning, feature engineering, regularization (L1/L2)

**Deep Learning**: CNNs, Transfer Learning, Data Augmentation, Batch Normalization, overfitting detection/mitigation

**Data Science**: EDA, statistical analysis, feature correlation, outlier handling, data leakage prevention

**Software Engineering**: REST API design (FastAPI), Docker containerization, Pydantic validation

**MLOps**: Model serialization, preprocessing artifacts, API-based serving
