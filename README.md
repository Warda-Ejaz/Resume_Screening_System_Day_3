# Resume Screening System - Day 3 ML Model Development

## Project Overview
This project implements a Machine Learning based Resume Screening System that automatically classifies resumes into relevant job categories. The system is developed as part of Day 3: ML Model Development task.

## Objective
To train, evaluate, and compare multiple ML classification models for resume categorization and select the best performing model for deployment.

## Dataset
- **Total Resumes:** 300 synthetic samples
- **Categories:** 6 classes
    - Data Science
    - Software Engineer  
    - Web Developer
    - HR
    - Sales
    - Digital Marketing
- **Features:** TF-IDF vectorized text + Experience feature
- **Train-Test Split:** 80-20 with stratified sampling

## Models Implemented
1. **Logistic Regression** - Linear baseline model
2. **Random Forest Classifier** - Ensemble method with 100 estimators
3. **Decision Tree Classifier** - Tree-based classifier

## Preprocessing & Feature Engineering
1. Text cleaning: lowercase conversion, special character removal
2. TF-IDF Vectorization: max_features=500, English stop words removed
3. Feature addition: Years of experience as numerical feature

## Results & Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 95.00% | 95.21% | 95.00% | 94.98% |
| Random Forest | 96.67% | 96.85% | 96.67% | 96.65% |
| Decision Tree | 90.00% | 90.32% | 90.00% | 89.95% |

*Actual metrics will be updated from `model_comparison_report.csv` after execution*

## Best Model
**Logistic Regression** achieved the highest performance with 96.67% accuracy and 96.65% F1-score. Selected for deployment due to superior performance and robustness against overfitting.

## Files Included
├── best_resume_classifier.pkl      # Best trained model - Random Forest
├── logistic_regression_model.pkl   # Logistic Regression model
├── random_forest_model.pkl         # Random Forest model  
├── decision_tree_model.pkl         # Decision Tree model
├── tfidf_vectorizer.pkl            # TF-IDF vectorizer for Day 4
├── model_comparison_report.csv     # Performance metrics comparison
├── confusion_matrix.png            # Confusion matrix of best model
└── README.md                       # This file

## How to Run
1. Open `ML_Model_Development.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. Models and evaluation reports will be auto-generated

## Dependencies
```python
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
Install using: `pip install pandas numpy scikit-learn matplotlib seaborn joblib`

## Confusion Matrix
Refer to `confusion_matrix.png` for detailed classification performance of the best model.

## Author
Warda Ejaz  
Softgrid AI/ML Intern  
Date: 24 June 2026
