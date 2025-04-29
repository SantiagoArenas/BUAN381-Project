# Load necessary libraries
library(tidyverse)
library(caret)
library(glmnet)
library(MASS)

# Load the datasets
train <- read.csv("../data/train.csv")
test <- read.csv("../data/test.csv")

# Ensure the target variable is numeric
train$latestPrice <- as.numeric(train$latestPrice)
test$latestPrice <- as.numeric(test$latestPrice)

# Partition the training data into training and validation sets
set.seed(123)
train_index <- createDataPartition(train$latestPrice, p = 0.7, list = FALSE)
train_data <- train[train_index, ]
valid_data <- train[-train_index, ]

# Define a function to calculate RMSE
calculate_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# 1. Simple Linear Regression
lm_model <- lm(latestPrice ~ livingAreaSqFt, data = train_data)
lm_predictions <- predict(lm_model, valid_data)
lm_rmse <- calculate_rmse(valid_data$latestPrice, lm_predictions)

# 2. Polynomial Regression (e.g., degree 2)
poly_model <- lm(latestPrice ~ poly(livingAreaSqFt, 2), data = train_data)
poly_predictions <- predict(poly_model, valid_data)
poly_rmse <- calculate_rmse(valid_data$latestPrice, poly_predictions)

# 3. Regularized Regression (Ridge and Lasso)
x_train <- model.matrix(latestPrice ~ ., train_data)[, -1]
y_train <- train_data$latestPrice
x_valid <- model.matrix(latestPrice ~ ., valid_data)[, -1]
y_valid <- valid_data$latestPrice

# Ridge Regression
ridge_model <- cv.glmnet(x_train, y_train, alpha = 0)
ridge_predictions <- predict(ridge_model, s = ridge_model$lambda.min, newx = x_valid)
ridge_rmse <- calculate_rmse(y_valid, ridge_predictions)

# Lasso Regression
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)
lasso_predictions <- predict(lasso_model, s = lasso_model$lambda.min, newx = x_valid)
lasso_rmse <- calculate_rmse(y_valid, lasso_predictions)

# 4. Multivariate Linear Regression
multi_lm_model <- lm(latestPrice ~ ., data = train_data)
multi_lm_predictions <- predict(multi_lm_model, valid_data)
multi_lm_rmse <- calculate_rmse(valid_data$latestPrice, multi_lm_predictions)

# 5. Generalized Additive Model (GAM)
library(mgcv)
gam_model <- gam(latestPrice ~ s(livingAreaSqFt) + s(yearBuilt), data = train_data)
gam_predictions <- predict(gam_model, valid_data)
gam_rmse <- calculate_rmse(valid_data$latestPrice, gam_predictions)

# Summarize Results
results <- data.frame(
  Model = c("Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression", "Multivariate Regression", "GAM"),
  RMSE = c(lm_rmse, poly_rmse, ridge_rmse, lasso_rmse, multi_lm_rmse, gam_rmse)
)

# Save results to a CSV file
write.csv(results, "../results/regression_results.csv", row.names = FALSE)

# Print results
print(results)