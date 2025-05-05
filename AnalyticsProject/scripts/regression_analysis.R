# Clear all variables and restart the session
rm(list = ls())
gc()  # Trigger garbage collection to free memory

# Load necessary libraries
library(tidyverse)
library(caret)
library(glmnet)
library(MASS)
library(mgcv)
library(e1071)
library(ggplot2)
library(corrplot)
library(dplyr)

# Load the datasets
train <- read.csv("AnalyticsProject/data/train.csv")
test <- read.csv("AnalyticsProject/data/test.csv")

# Ensure the target variable is numeric
train$latestPrice <- as.numeric(train$latestPrice)
test$latestPrice <- as.numeric(test$latestPrice)

# Partition the training data into training and validation sets
set.seed(123)
train_index <- createDataPartition(train$latestPrice, p = 0.7, list = FALSE)
train_data <- train[train_index, ]
valid_data <- train[-train_index, ]

# Remove the latest_saledate and latestPriceSource variables from train_data and valid_data
# Remove the latest_saledate and latestPriceSource variables
train_data <- train_data %>% dplyr::select(-latest_saledate, -latestPriceSource)
valid_data <- valid_data %>% dplyr::select(-latest_saledate, -latestPriceSource)

# Initialize an empty data frame to store results
results <- data.frame(
  Model = character(),
  In_Sample_RMSE = numeric(),
  Out_Sample_RMSE = numeric(),
  R2 = numeric(),
  stringsAsFactors = FALSE
)

# Correlation matrix
numeric_vars <- train %>% select_if(is.numeric)
cor_matrix <- cor(numeric_vars, use = "complete.obs")
#print(cor_matrix)

# Save the correlation matrix as a CSV
write.csv(cor_matrix, "AnalyticsProject/results/correlation_matrix.csv")

# Define a function to calculate RMSE
calculate_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Define a function to calculate R²
calculate_r2 <- function(actual, predicted) {
  1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
}

# 1. Simple Linear Regression
lm_model <- lm(latestPrice ~ livingAreaSqFt, data = train_data)
lm_in_sample_predictions <- predict(lm_model, train_data)
lm_out_sample_predictions <- predict(lm_model, valid_data)
lm_in_sample_rmse <- calculate_rmse(train_data$latestPrice, lm_in_sample_predictions)
lm_out_sample_rmse <- calculate_rmse(valid_data$latestPrice, lm_out_sample_predictions)
lm_r2 <- calculate_r2(valid_data$latestPrice, lm_out_sample_predictions)

# Append results for Linear Regression
results <- rbind(results, data.frame(
  Model = "Linear Regression",
  In_Sample_RMSE = lm_in_sample_rmse,
  Out_Sample_RMSE = lm_out_sample_rmse,
  R2 = lm_r2
))

# Remove variables associated with Linear Regression
rm(lm_in_sample_predictions, lm_out_sample_predictions, lm_in_sample_rmse, lm_out_sample_rmse, lm_r2)
gc()  # Trigger garbage collection to free memory

# 2. Polynomial Regression (e.g., degree 2)
poly_model <- lm(latestPrice ~ poly(livingAreaSqFt, 2), data = train_data)
poly_in_sample_predictions <- predict(poly_model, train_data)
poly_out_sample_predictions <- predict(poly_model, valid_data)
poly_in_sample_rmse <- calculate_rmse(train_data$latestPrice, poly_in_sample_predictions)
poly_out_sample_rmse <- calculate_rmse(valid_data$latestPrice, poly_out_sample_predictions)
poly_r2 <- calculate_r2(valid_data$latestPrice, poly_out_sample_predictions)

# Append results for Polynomial Regression
results <- rbind(results, data.frame(
  Model = "Polynomial Regression",
  In_Sample_RMSE = poly_in_sample_rmse,
  Out_Sample_RMSE = poly_out_sample_rmse,
  R2 = poly_r2
))

# Remove variables associated with Polynomial Regression
rm(poly_in_sample_predictions, poly_out_sample_predictions, poly_in_sample_rmse, poly_out_sample_rmse, poly_r2)
gc()

# 3. Regularized Regression (Ridge and Lasso)
# Match factor levels between train and valid datasets
for (col in names(train_data)) {
  if (is.factor(train_data[[col]])) {
    valid_data[[col]] <- factor(valid_data[[col]], levels = levels(train_data[[col]]))
  }
}

# Now this won't fail
# Align factor levels between train_data and valid_data
for (col in names(train_data)) {
  if (is.factor(train_data[[col]])) {
    valid_data[[col]] <- factor(valid_data[[col]], levels = levels(train_data[[col]]))
  }
}

# Build model matrices
formula <- latestPrice ~ .
x_train <- model.matrix(formula, train_data)[, -1]  # Remove intercept column
x_valid_tmp <- model.matrix(formula, valid_data)[, -1]

# Ensure x_valid_tmp is a numeric matrix
x_valid_tmp <- as.matrix(x_valid_tmp)

# Add missing columns to x_valid_tmp
missing_cols <- setdiff(colnames(x_train), colnames(x_valid_tmp))
for (col in missing_cols) {
  x_valid_tmp <- cbind(x_valid_tmp, setNames(matrix(0, nrow = nrow(x_valid_tmp), ncol = 1), col))
}

missing_in_valid <- setdiff(colnames(x_train), colnames(x_valid_tmp))
extra_in_valid <- setdiff(colnames(x_valid_tmp), colnames(x_train))

# Print the differences
print(paste("Missing in x_valid_tmp:", paste(missing_in_valid, collapse = ", ")))
print(paste("Extra in x_valid_tmp:", paste(extra_in_valid, collapse = ", ")))

# Remove extra columns from x_valid_tmp
x_valid_tmp <- x_valid_tmp[, colnames(x_train), drop = FALSE]

# Reorder columns to match x_train
x_valid <- x_valid_tmp[, colnames(x_train), drop = FALSE]

# Define the response variable
y_train <- train_data$latestPrice
y_valid <- valid_data$latestPrice

# Ridge Regression using glmnet
ridge_model <- cv.glmnet(x_train, y_train, alpha = 0)  # alpha = 0 for Ridge Regression

# Predictions
ridge_in_sample_predictions <- predict(ridge_model, s = ridge_model$lambda.min, newx = x_train)
str(x_valid)  # Check the structure of x_valid
ridge_out_sample_predictions <- predict(ridge_model, s = ridge_model$lambda.min, newx = x_valid)

# Calculate performance metrics
ridge_in_sample_rmse <- calculate_rmse(y_train, ridge_in_sample_predictions)
ridge_out_sample_rmse <- calculate_rmse(y_valid, ridge_out_sample_predictions)
ridge_r2 <- calculate_r2(y_valid, ridge_out_sample_predictions)

# Append results for Ridge Regression
results <- rbind(results, data.frame(
  Model = "Ridge Regression",
  In_Sample_RMSE = ridge_in_sample_rmse,
  Out_Sample_RMSE = ridge_out_sample_rmse,
  R2 = ridge_r2
))

# Remove variables associated with Ridge Regression
rm(ridge_in_sample_predictions, ridge_out_sample_predictions, ridge_in_sample_rmse, ridge_out_sample_rmse, ridge_r2)
gc()

# Lasso Regression
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)
lasso_in_sample_predictions <- predict(lasso_model, s = lasso_model$lambda.min, newx = x_train)
lasso_out_sample_predictions <- predict(lasso_model, s = lasso_model$lambda.min, newx = x_valid)
lasso_in_sample_rmse <- calculate_rmse(y_train, lasso_in_sample_predictions)
lasso_out_sample_rmse <- calculate_rmse(y_valid, lasso_out_sample_predictions)
lasso_r2 <- calculate_r2(y_valid, lasso_out_sample_predictions)

# Append results for Lasso Regression
results <- rbind(results, data.frame(
  Model = "Lasso Regression",
  In_Sample_RMSE = lasso_in_sample_rmse,
  Out_Sample_RMSE = lasso_out_sample_rmse,
  R2 = lasso_r2
))

# Remove variables associated with Lasso Regression
rm(lasso_in_sample_predictions, lasso_out_sample_predictions, lasso_in_sample_rmse, lasso_out_sample_rmse, lasso_r2)
gc()

# 4. Multivariate Linear Regression
multi_model <- lm(latestPrice ~ ., data = train_data)
multi_in_pred <- predict(multi_model, train_data)
multi_out_pred <- predict(multi_model, valid_data)

multi_in_rmse <- calculate_rmse(train_data$latestPrice, multi_in_pred)
multi_out_rmse <- calculate_rmse(valid_data$latestPrice, multi_out_pred)
multi_r2 <- calculate_r2(valid_data$latestPrice, multi_out_pred)

# Append results for Multivariate Linear Regression
results <- rbind(results, data.frame(
  Model = "Multivariate Linear Regression",
  In_Sample_RMSE = multi_in_rmse,
  Out_Sample_RMSE = multi_out_rmse,
  R2 = multi_r2
))

# Remove variables associated with Multivariate Linear Regression
rm(multi_in_pred, multi_out_pred, multi_in_rmse, multi_out_rmse, multi_r2)
gc()


# 5. Generalized Additive Model (GAM)
gam_model <- gam(latestPrice ~ s(livingAreaSqFt) + s(yearBuilt), data = train_data)
gam_in_sample_predictions <- predict(gam_model, train_data)
gam_out_sample_predictions <- predict(gam_model, valid_data)
gam_in_sample_rmse <- calculate_rmse(train_data$latestPrice, gam_in_sample_predictions)
gam_out_sample_rmse <- calculate_rmse(valid_data$latestPrice, gam_out_sample_predictions)
gam_r2 <- calculate_r2(valid_data$latestPrice, gam_out_sample_predictions)
# Append results for GAM
results <- rbind(results, data.frame(
  Model = "GAM",
  In_Sample_RMSE = gam_in_sample_rmse,
  Out_Sample_RMSE = gam_out_sample_rmse,
  R2 = gam_r2
))
# Remove variables associated with GAM
rm(gam_in_sample_predictions, gam_out_sample_predictions, gam_in_sample_rmse, gam_out_sample_rmse, gam_r2)
gc()

# Plot all bivariate models
png("AnalyticsProject/results/img/all_models_comparison.png", width = 1200, height = 800, res = 150)
ggplot(train_data, aes(x = livingAreaSqFt, y = latestPrice)) +
  geom_point(alpha = 0.6, color = "gray") +
  geom_smooth(method = "lm", color = "blue", se = FALSE, aes(linetype = "Linear")) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), color = "red", se = FALSE, aes(linetype = "Polynomial")) +
  geom_line(aes(y = predict(ridge_model, s = ridge_model$lambda.min, newx = x_train), linetype = "Ridge"), color = "green") +
  geom_line(aes(y = predict(lasso_model, s = lasso_model$lambda.min, newx = x_train), linetype = "Lasso"), color = "purple") +
  geom_line(aes(y = predict(multi_model, train_data), linetype = "Multivariate"), color = "orange") +
  geom_line(aes(y = predict(gam_model, train_data), linetype = "GAM"), color = "brown") +
  labs(title = "Comparison of All Models",
       x = "Living Area (SqFt)",
       y = "Latest Price") +
  theme_minimal() +
  scale_linetype_manual(name = "Model", 
                        values = c("Linear" = "solid", 
                                   "Polynomial" = "dashed", 
                                   "Ridge" = "dotdash", 
                                   "Lasso" = "twodash", 
                                   "Multivariate" = "longdash", 
                                   "GAM" = "dotted"))
dev.off()

# Support Vector Machine (SVM)
svm_model <- svm(latestPrice ~ ., data = train_data, kernel = "linear")
svm_in_sample_predictions <- predict(svm_model, train_data)
svm_out_sample_predictions <- predict(svm_model, valid_data)
svm_in_sample_rmse <- calculate_rmse(train_data$latestPrice, svm_in_sample_predictions)
svm_out_sample_rmse <- calculate_rmse(valid_data$latestPrice, svm_out_sample_predictions)
svm_r2 <- calculate_r2(valid_data$latestPrice, svm_out_sample_predictions)
# Append results for SVM
results <- rbind(results, data.frame(
  Model = "SVM",
  In_Sample_RMSE = svm_in_sample_rmse,
  Out_Sample_RMSE = svm_out_sample_rmse,
  R2 = svm_r2
))
# Remove variables associated with SVM
rm(svm_in_sample_predictions, svm_out_sample_predictions, svm_in_sample_rmse, svm_out_sample_rmse, svm_r2)
gc()

# Random Forest
library(randomForest)
rf_model <- randomForest(latestPrice ~ ., data = train_data, ntree = 100)
rf_in_sample_predictions <- predict(rf_model, train_data)
rf_out_sample_predictions <- predict(rf_model, valid_data)
rf_in_sample_rmse <- calculate_rmse(train_data$latestPrice, rf_in_sample_predictions)
rf_out_sample_rmse <- calculate_rmse(valid_data$latestPrice, rf_out_sample_predictions)
rf_r2 <- calculate_r2(valid_data$latestPrice, rf_out_sample_predictions)
# Plot Random Forest
png("AnalyticsProject/results/img/random_forest.png", width = 1200, height = 800, res = 150)
ggplot(train_data, aes(x = livingAreaSqFt, y = latestPrice)) +
  geom_point(alpha = 0.6, color = "gray", aes(shape = "Data Points")) +
  geom_smooth(method = "rf", color = "green", se = FALSE, aes(linetype = "Random Forest")) +
  labs(title = "Random Forest Model",
       x = "Living Area (SqFt)",
       y = "Latest Price") +
  theme_minimal() +
  scale_shape_manual(name = "Legend", values = c("Data Points" = 16)) +
  scale_linetype_manual(name = "Legend", values = c("Random Forest" = "solid"))
dev.off()
# Append results for Random Forest
results <- rbind(results, data.frame(
  Model = "Random Forest",
  In_Sample_RMSE = rf_in_sample_rmse,
  Out_Sample_RMSE = rf_out_sample_rmse,
  R2 = rf_r2
))
# Remove variables associated with Random Forest
rm(rf_in_sample_predictions, rf_out_sample_predictions, rf_in_sample_rmse, rf_out_sample_rmse, rf_r2)
gc()

# Save the final results table
write.csv(results, "AnalyticsProject/results/final_regression_results.csv", row.names = FALSE)
