# Analytics Project Scripts

This directory contains the scripts used for the regression and classification analyses in the Analytics Project. Below is a summary of the scripts and their purposes.

---

## **Scripts Overview**

### **1. `regression_analysis.R`**
- **Purpose**: Performs regression analysis to predict `latestPrice` based on features such as `livingAreaSqFt`, `yearBuilt`, and others.
- **Key Models**:
  - Linear Regression
  - Polynomial Regression
  - Ridge Regression
  - Lasso Regression
  - Multivariate Linear Regression
  - Generalized Additive Model (GAM)
- **Outputs**:
  - R² values for each model.
  - Visualizations comparing model predictions (e.g., `all_models_comparison.png` and `filtered_models_comparison.png`).

### **2. `classification_analysis.R`**
- **Purpose**: Performs classification analysis to predict `homeType` (e.g., Single Family, Condo) based on features such as `livingAreaSqFt`, `yearBuilt`, and others.
- **Key Models**:
  - Logistic Regression
  - Probit Model
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- **Outputs**:
  - Accuracy and AUC for each model.
  - Visualizations such as ROC curves and variable importance plots.

---

## **How to Run the Scripts**

1. **Set Up the Environment**:
   - Ensure all required libraries are installed:
     ```r
     install.packages(c("caret", "glmnet", "MASS", "mgcv", "e1071", "ggplot2", "corrplot", "rpart.plot", "randomForest", "pROC"))
     ```

2. **Run the Scripts**:
   - Execute the scripts in R or RStudio:
     ```r
     source("regression_analysis.R")
     source("classification_analysis.R")
     ```

3. **Outputs**:
   - Results and visualizations will be saved in the `results` directory.

---

## **Dependencies**
The following R libraries are required to run the scripts:
- `caret`
- `glmnet`
- `MASS`
- `mgcv`
- `e1071`
- `ggplot2`
- `corrplot`
- `rpart.plot`
- `randomForest`
- `pROC`

---

## **Directory Structure**