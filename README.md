#  Telcom Customer Churn Prediction with MLflow

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/mlflow-2.0+-green.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A comprehensive machine learning project for predicting customer churn in telecommunications using MLflow for experiment tracking, model management, and deployment.

##  Project Overview

This project implements a complete MLflow-powered machine learning pipeline to predict customer churn for a telecommunications company. The solution includes data preprocessing, feature engineering, model training with multiple algorithms, hyperparameter optimization, and production deployment patterns.

###  Key Features

- **Complete ML Pipeline**: End-to-end churn prediction workflow
- **MLflow Integration**: Comprehensive experiment tracking and model management
- **Multiple Algorithms**: Random Forest, XGBoost, LightGBM, and more
- **Advanced Analytics**: Feature importance, SHAP analysis, and performance metrics
- **Production Ready**: Model registry, staging, and deployment patterns
- **Interactive Dashboard**: MLflow UI for experiment comparison and monitoring

## Project Structure

```
Churn_Prediction/
│
├── Telco_Customer_Churn.csv           # Original dataset
├── cleaned_telcom_data.csv           # Processed dataset
├── customer_id_mapping.csv           # Customer ID mappings
│
├── churn_eda.ipynb            # Exploratory Data Analysis
├── churn_model_training.ipynb # Complete MLflow training pipeline
│
├── mlruns/                           # MLflow experiment tracking data
├── artifacts/                        # MLflow artifacts and models
├── models/                           # Saved model files
│
├── .gitignore                        # Git ignore rules
└── README.md                         # Project documentation
```

## Quick Start

### Prerequisites

- Python 3.8+
- pip or conda package manager
- Git (optional, for version control)

### 1. Environment Setup

```bash
# Clone the repository (if using git)
git clone https://github.com/TumainiC/Churn_Prediction.git
cd Churn_Prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Install Dependencies

```bash
# Core ML packages
pip install pandas numpy scikit-learn xgboost lightgbm

# MLflow and tracking
pip install mlflow

# Visualization and analysis
pip install matplotlib seaborn plotly shap

# Jupyter notebook
pip install jupyter ipykernel
```

### 3. Launch MLflow UI

```bash
# Start MLflow tracking server
mlflow ui

# Access the dashboard at: http://localhost:5000
```

### 4. Run the Analysis

```bash
# Start Jupyter notebook
jupyter notebook

# Open and run:
# - churn_eda.ipynb (for data exploration)
# - churn_model_training.ipynb (for complete MLflow pipeline)
```

## Dataset Information

### Dataset Overview
- **Source**: Telco Customer Churn Dataset
- **Records**: ~7,000 customers
- **Features**: 21 attributes including demographics, services, and account information
- **Target**: Binary churn prediction (Yes/No)

### Key Features
- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Services**: Phone, Internet, Streaming, Security, Support
- **Account**: Contract type, Payment method, Tenure, Charges
- **Target**: Churn (customer left in the last month)

## MLflow Integration

### Experiment Tracking
- **Parameters**: Algorithm hyperparameters, preprocessing settings
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Artifacts**: Model files, plots, feature importance, confusion matrices
- **Tags**: Model type, environment, team, version

### Model Registry
- **Staging Pipeline**: None → Staging → Production
- **Version Management**: Automatic versioning and lineage tracking
- **Model Metadata**: Performance metrics, training data info, model signatures

### Production Deployment
- **Batch Scoring**: Large-scale prediction pipelines
- **Real-time Inference**: Individual customer risk assessment
- **Monitoring**: Performance tracking and drift detection
- **Alerting**: Automated notifications for model degradation

## Model Performance

### Algorithms Tested
| Algorithm | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-----------|----------|-----------|---------|----------|---------|
| Logistic Regression (Tuned) | 0.8087 | 0.6598 | 0.5750 | 0.6145 | 0.8476 |
| Logistic Regression | 0.8087 | 0.6598 | 0.5750 | 0.6145 | 0.8469 |
| Random Forest | 0.7973 | 0.6422 | 0.5321 | 0.5820 | 0.8377 |
| XGBoost | 0.7831 | 0.6041 | 0.5286 | 0.5638 | 0.8383 |
| LightGBM | 0.7860 | 0.6098 | 0.5357 | 0.5703 | 0.8442 |

*Results based on validation set performance. Models ranked by weighted score (60% Recall + 40% AUC-ROC).*

### Key Insights
- **Top Predictors**: Contract type, tenure, monthly charges, internet service
- **Churn Patterns**: Month-to-month contracts have highest churn rates
- **Feature Importance**: Financial factors (charges) and service tenure are critical
- **Model Recommendation**: Logistic Regression (Tuned) - Best balance of recall and AUC-ROC
- **Best Churn Detection**: Logistic Regression achieves 57.5% recall with 84.8% AUC-ROC
- **Weighted Performance**: Tuned model achieves 0.6840 weighted score (60% Recall + 40% AUC-ROC)

## MLflow Commands Reference

### Basic Operations
```bash
# Start MLflow UI
mlflow ui

# Start with custom port
mlflow ui --port 5001

# Start MLflow server with backend
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./artifacts
```

### Model Management
```python
# Register model
mlflow.sklearn.log_model(model, "churn_model", registered_model_name="ChurnPredictor")

# Load model for inference
model = mlflow.pyfunc.load_model("models:/ChurnPredictor/Production")

# Make predictions
predictions = model.predict(new_data)
```

## Monitoring and Alerting

### Performance Monitoring
- **Metrics Tracking**: Real-time accuracy, precision, recall monitoring
- **Threshold Alerts**: Automated notifications when performance drops
- **Data Drift Detection**: Statistical tests for input data changes
- **Dashboard Updates**: Live performance visualization

### Production Checklist
- [ ] Model registered in MLflow Model Registry
- [ ] Staging environment testing completed
- [ ] Production deployment approved
- [ ] Monitoring dashboard configured
- [ ] Alert thresholds defined
- [ ] Rollback procedure documented

## Development Workflow

### 1. Data Exploration
Run `churn_eda.ipynb` to understand the dataset, identify patterns, and perform initial feature analysis.

### 2. Model Development
Use `churn_model_training.ipynb` to:
- Train multiple algorithms
- Track experiments with MLflow
- Compare model performance
- Register best models

### 3. Model Deployment
- Promote models through registry stages
- Set up monitoring and alerting
- Deploy to production environment
- Monitor performance and retrain as needed

## Additional Resources

### MLflow Documentation
- [MLflow Official Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

### Machine Learning Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact the development team, aka me.

---

**Happy Machine Learning with MLflow!**

*Last updated: October 21, 2025*