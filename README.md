# Student Depression ML Pipeline

## Description
This project implements a machine learning pipeline to predict student depression risk using XGBoost classification.

## Dataset
- **Size:** 27,901 student records
- **Features:** 16 features including academic pressure, sleep duration, financial stress, and more
- **Target:** Depression (Binary Classification: 0 = No Risk, 1 = At Risk)

## Model Performance
- **Algorithm:** XGBoost Classifier
- **ROC-AUC Score:** 0.905
- **Optimal Hyperparameters:**
  - Learning Rate: 0.01
  - Max Depth: 5
  - N Estimators: 100

## Key Findings
- Suicidal thoughts history is the strongest predictor
- Academic pressure and financial stress are significant factors
- Sleep duration patterns show correlation with depression risk

## Files Included
- `depression_ml_pipeline.py` - Main Python script
- `student_depression_dataset.csv` - Dataset
- `confusion_matrix_evaluation.png` - Confusion matrix visualization
- `feature_importance_analysis.png` - Feature importance plot
- `roc_curve_evaluation.png` - ROC curve analysis
- `academic_pressure_vs_depression.png` - Academic pressure visualization
- `sleep_duration_vs_depression.png` - Sleep duration visualization

## How to Use
1. Ensure you have Python and required libraries installed
2. Run the script: `python depression_ml_pipeline.py`
3. The model will train and generate evaluation plots

## Author
PETER OKONKWO
Date: May 2026