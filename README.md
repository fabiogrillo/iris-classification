# Machine Learning Projects

This repository contains multiple machine learning projects demonstrating various ML techniques, algorithms, and deployment strategies.

## Projects

### 1. Iris Classification Project
**Status**: ✅ Complete
**Type**: Multi-class Classification
**Location**: [`Iris-Project/`](./Iris-Project/)

**Description**: End-to-end machine learning pipeline for classifying iris flower species using scikit-learn, with a FastAPI REST API deployment.

**Key Features**:
- Multi-class classification (3 species)
- Model comparison (SVM, Logistic Regression, Random Forest, Decision Tree)
- 96.7% test accuracy with Support Vector Machine
- RESTful API with FastAPI and Pydantic validation
- Docker containerization for deployment
- Comprehensive EDA and feature engineering

**Tech Stack**: Python, scikit-learn, FastAPI, Docker, Pandas, Matplotlib, Seaborn

**Documentation**: See [Iris-Project/README.md](./Iris-Project/README.md) for detailed project documentation

---

## Repository Structure

```
MachineLearning/
├── Iris-Project/           # Iris flower classification project
│   ├── app.py             # FastAPI application
│   ├── Dockerfile         # Container configuration
│   ├── iris_classification.ipynb  # Jupyter notebook with EDA and training
│   ├── models/            # Serialized ML models
│   ├── figures/           # EDA visualizations
│   ├── requirements.txt   # Python dependencies
│   └── README.md          # Project documentation
│
└── README.md              # This file (project index)
```

---

## Future Projects

Planned projects to be added:
- **House Price Prediction**: Regression analysis with feature engineering
- **Sentiment Analysis**: NLP with transformer models
- **Image Classification**: Deep learning with CNNs
- **Recommender System**: Collaborative filtering implementation

---

## How to Use This Repository

Each project is self-contained in its own directory with:
- Complete source code
- Jupyter notebooks with analysis
- Model artifacts
- Docker deployment files (where applicable)
- Comprehensive README documentation

Navigate to individual project directories for specific setup and usage instructions.

---

## Skills Demonstrated

Across these projects, I demonstrate proficiency in:

**Machine Learning**:
- Supervised learning (classification, regression)
- Model selection and evaluation
- Cross-validation and hyperparameter tuning
- Feature engineering and preprocessing

**Data Science**:
- Exploratory Data Analysis (EDA)
- Statistical analysis and visualization
- Feature correlation and selection
- Data leakage prevention

**Software Engineering**:
- API design and development (FastAPI, RESTful principles)
- Containerization (Docker)
- Code quality and documentation
- Version control (Git)

**MLOps**:
- Model serialization and versioning
- Deployment strategies
- API-based model serving
- Monitoring and evaluation

---

**Contact**: For questions or collaboration opportunities, please reach out!
