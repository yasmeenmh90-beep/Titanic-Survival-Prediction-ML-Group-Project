
# Titanic Survival Prediction — ML Group Project

A complete machine learning lifecycle project predicting passenger survival on the RMS Titanic, using the Kaggle Titanic dataset. Built as part of an Introduction to Machine Learning module.

## Team

This was a **group project** completed by:
- Ali Ahmar
- Yasmeen Azmat Ali
- Shaza

## Project Overview

The Titanic dataset (891 passengers) is used to predict survival based on features like class, gender, age, fare, and family size. The project follows a full ML lifecycle: exploratory data analysis, data preprocessing, feature engineering, model development, and evaluation — with attention to dataset biases (gender, age, class).

---

## Approach

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of Age, Fare, Class, Sex, Embarked
- Missing value visualization (`missingno`)
- Survival rate breakdowns by gender, class, and age group

### 2. Data Preparation
- Handled missing values (Age, Cabin, Embarked)
- Feature engineering: created `FamilySize` from `SibSp` + `Parch`
- Encoded categorical variables, scaled numerical features
- Addressed class imbalance using `imbalanced-learn`

### 3. Model Development
Trained and compared multiple classifiers:

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 86.82% |
| Decision Tree | 81.82% |
| Random Forest | 87.27% |
| **Random Forest (Top 10 Features)** | **88.64%** |
| Gradient Boosting | 86.36% |
| SVM (hyperparameter tuned) | 88.18% |
| SVM (balanced class weights) | 86.36% |

- Performed cross-validation (5-fold) for robustness checking
- Feature importance analysis to identify top predictors
- Hyperparameter tuning via grid search

---

## Key Findings

- **Random Forest with the top 10 features** gave the best test accuracy (88.64%), showing that feature selection improved generalization over using all features
- Gender, passenger class, and fare were consistently the strongest predictors of survival
- Cross-validation scores stayed consistent (~82-83%) across models, indicating reasonable generalization without severe overfitting

---

## Technologies Used

- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, Decision Trees, Random Forest, SVM, Gradient Boosting)
- Matplotlib, Seaborn (visualization)
- ydata-profiling (automated EDA reports)
- missingno (missing data visualization)
- imbalanced-learn (class imbalance handling)

---

## Files

- `Y_Final_MLProject.ipynb` — full notebook: EDA, preprocessing, model training, evaluation

---

## License

This project is for academic/portfolio purposes.
