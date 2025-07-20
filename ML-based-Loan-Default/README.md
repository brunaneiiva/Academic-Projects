# 📊 Loan Default Prediction using CRISP-DM

This project applies Data Mining and Machine Learning techniques to predict whether a bank customer will default on a loan. It was developed as part of the *Data Analysis and Knowledge Management* course in the Master's in Industrial Engineering and Management at ISEP (Instituto Superior de Engenharia do Porto).

The methodology follows the CRISP-DM process, which includes six key stages: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment (not implemented in this academic context).

---

## 🔍 Project Overview

- **Dataset**: Bank loan dataset with 67,000+ rows and 35 features (both numeric and categorical).
- **Goal**: Predict whether a customer will fail to repay a loan.
- **Challenge**: The dataset is highly imbalanced, which impacts model performance.

---

## 🧠 Steps and Techniques Used

### 1. Business Understanding
- Defined the business problem: reduce financial risk by identifying potential loan defaults.

### 2. Data Understanding
- Explored variable types, missing values, and duplicates.
- Generated histograms, boxplots, and a correlation matrix.
- Performed Chi-Square tests to detect associations between categorical variables.

### 3. Data Preparation
- Removed irrelevant or constant-value variables (e.g., ID, Payment Plan).
- Scaled numeric variables using `StandardScaler`.
- Encoded categorical variables with `LabelEncoder`.

### 4. Modeling
Implemented and compared several classification algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Neural Networks (MLP)
- Naive Bayes

### 5. Evaluation
- Used confusion matrix, precision, recall, F1-score, ROC curve, and AUC to evaluate performance.
- Highlighted the low recall for the minority class (loan defaulters) and the limitations of accuracy as a metric in imbalanced datasets.


## 📁 Files

- `andoc.py`: Main Python script with all steps from data processing to model evaluation.
- `TRABALHO_FINAL_ANDOC.pdf`: Full report (in Portuguese) with explanation of the CRISP-DM methodology and results.
- `figures/`: Contains generated visualizations like histograms, boxplots, and ROC curves.

---

## 🚀 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Scipy
