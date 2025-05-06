# Clear all variables and restart the session
rm(list = ls())
gc()

# Load necessary libraries
library(tidyverse)
library(caret)
library(pROC)
library(e1071)
library(rpart)
library(randomForest)
library(ggplot2)
library(dplyr)

# Load the dataset
train <- read.csv("AnalyticsProject/data/train.csv")
test <- read.csv("AnalyticsProject/data/test.csv")

# Convert homeType back to its original form if it was one-hot encoded
if ("homeType_Single.Family" %in% colnames(train)) {
  train$homeType <- ifelse(train$homeType_Single.Family == 1, "Single Family", "Other")
  train <- train %>% dplyr::select(-homeType_Single.Family, -homeType_Condo, -homeType_Multiple.Occupancy, -homeType_Townhouse, -homeType_Vacant.Land)
}

if ("homeType_Single.Family" %in% colnames(test)) {
  test$homeType <- ifelse(test$homeType_Single.Family == 1, "Single Family", "Other")
  test <- test %>% dplyr::select(-homeType_Single.Family, -homeType_Condo, -homeType_Multiple.Occupancy, -homeType_Townhouse, -homeType_Vacant.Land)
}

# Check the distribution of the target variable
print(table(train$homeType))

# Initialize a results data frame
results <- data.frame(
  Model = character(),
  InSampleAccuracy = numeric(),
  OutOfSampleAccuracy = numeric(),
  AUC = numeric(),
  stringsAsFactors = FALSE
)

# Split the training dataset into training and validation sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(train$homeType, p = 0.7, list = FALSE)
train_data <- train[train_index, ]
valid_data <- train[-train_index, ]

# Remove unnecessary variables using dplyr::select
train_data <- train_data %>% dplyr::select(-latest_saledate, -latestPriceSource)
valid_data <- valid_data %>% dplyr::select(-latest_saledate, -latestPriceSource)

# Ensure the target variable is properly encoded as 0 and 1
train_data$homeType <- ifelse(train_data$homeType == "Single Family", 1, 0)
valid_data$homeType <- ifelse(valid_data$homeType == "Single Family", 1, 0)

# Logistic Regression
logit_model <- glm(homeType ~ ., data = train_data, family = binomial)

# In-sample predictions
logit_in_sample_predictions <- predict(logit_model, train_data, type = "response")
logit_in_sample_class <- ifelse(logit_in_sample_predictions > 0.5, 1, 0)
logit_in_sample_confusion <- confusionMatrix(as.factor(logit_in_sample_class), as.factor(train_data$homeType))

# Out-of-sample predictions
logit_out_sample_predictions <- predict(logit_model, valid_data, type = "response")
logit_out_sample_class <- ifelse(logit_out_sample_predictions > 0.5, 1, 0)
logit_out_sample_confusion <- confusionMatrix(as.factor(logit_out_sample_class), as.factor(valid_data$homeType))

# Calculate AUC using probabilities
logit_in_sample_auc <- roc(train_data$homeType, logit_in_sample_predictions)$auc
logit_out_sample_auc <- roc(valid_data$homeType, logit_out_sample_predictions)$auc

# Save Logistic Regression ROC Curve (Out-of-sample)
png("AnalyticsProject/results/img/class/logistic_regression_roc.png", width = 800, height = 600)
plot(roc(valid_data$homeType, logit_out_sample_predictions), main = paste("Logistic Regression ROC Curve (AUC =", round(logit_out_sample_auc, 3), ")"))
dev.off()

# Append results for Logistic Regression
results <- rbind(results, data.frame(
  Model = "Logistic Regression",
  InSampleAccuracy = logit_in_sample_confusion$overall["Accuracy"],
  OutOfSampleAccuracy = logit_out_sample_confusion$overall["Accuracy"],
  AUC = logit_out_sample_auc
))

# Probit Model
probit_model <- glm(homeType ~ ., data = train_data, family = binomial(link = "probit"))

# In-sample predictions
probit_in_sample_predictions <- predict(probit_model, train_data, type = "response")
probit_in_sample_class <- ifelse(probit_in_sample_predictions > 0.5, 1, 0)
probit_in_sample_confusion <- confusionMatrix(as.factor(probit_in_sample_class), as.factor(train_data$homeType))

# Out-of-sample predictions
probit_out_sample_predictions <- predict(probit_model, valid_data, type = "response")
probit_out_sample_class <- ifelse(probit_out_sample_predictions > 0.5, 1, 0)
probit_out_sample_confusion <- confusionMatrix(as.factor(probit_out_sample_class), as.factor(valid_data$homeType))

# Calculate AUC using probabilities
probit_in_sample_auc <- roc(train_data$homeType, probit_in_sample_predictions)$auc
probit_out_sample_auc <- roc(valid_data$homeType, probit_out_sample_predictions)$auc

# Save Probit Model ROC Curve (Out-of-sample)
png("AnalyticsProject/results/img/class/probit_model_roc.png", width = 800, height = 600)
plot(roc(valid_data$homeType, probit_out_sample_predictions), main = paste("Probit Model ROC Curve (AUC =", round(probit_out_sample_auc, 3), ")"))
dev.off()

# Append results for Probit Model
results <- rbind(results, data.frame(
  Model = "Probit Model",
  InSampleAccuracy = probit_in_sample_confusion$overall["Accuracy"],
  OutOfSampleAccuracy = probit_out_sample_confusion$overall["Accuracy"],
  AUC = probit_out_sample_auc
))

# Decision Tree
tree_model <- rpart(homeType ~ ., data = train_data, method = "class")

# In-sample predictions
tree_in_sample_predictions <- predict(tree_model, train_data, type = "class")
tree_in_sample_confusion <- confusionMatrix(as.factor(tree_in_sample_predictions), as.factor(train_data$homeType))

# Out-of-sample predictions
tree_out_sample_predictions <- predict(tree_model, valid_data, type = "class")
tree_out_sample_confusion <- confusionMatrix(as.factor(tree_out_sample_predictions), as.factor(valid_data$homeType))

# Save Decision Tree Plot with Enhanced Visualization
png("AnalyticsProject/results/img/class/decision_tree.png", width = 800, height = 600)
rpart.plot::rpart.plot(tree_model, type = 3, extra = 101, under = TRUE, fallen.leaves = TRUE, 
             main = "Decision Tree Visualization", cex = 0.8)
dev.off()

# Append results for Decision Tree
results <- rbind(results, data.frame(
  Model = "Decision Tree",
  InSampleAccuracy = tree_in_sample_confusion$overall["Accuracy"],
  OutOfSampleAccuracy = tree_out_sample_confusion$overall["Accuracy"],
  AUC = NA  # AUC not applicable for Decision Tree
))

# Recode homeType as factor for classification
train_data$homeType <- as.factor(train_data$homeType)
valid_data$homeType <- as.factor(valid_data$homeType)

# Refit the Random Forest model as a classification model
rf_model <- randomForest(homeType ~ ., data = train_data, ntree = 50)

# In-sample predictions
rf_in_sample_predictions <- predict(rf_model, train_data)
rf_in_sample_confusion <- confusionMatrix(rf_in_sample_predictions, train_data$homeType)

# Out-of-sample predictions
rf_out_sample_predictions <- predict(rf_model, valid_data)
rf_out_sample_confusion <- confusionMatrix(rf_out_sample_predictions, valid_data$homeType)

# Predict probabilities for class "1"
rf_in_sample_probabilities <- predict(rf_model, train_data, type = "prob")[, "1"]
rf_out_sample_probabilities <- predict(rf_model, valid_data, type = "prob")[, "1"]

# Calculate AUC
rf_in_sample_auc <- roc(as.numeric(train_data$homeType), rf_in_sample_probabilities)$auc
rf_out_sample_auc <- roc(as.numeric(valid_data$homeType), rf_out_sample_probabilities)$auc

# Save Random Forest Variable Importance Plot
png("AnalyticsProject/results/img/class/random_forest_importance.png", width = 800, height = 600)
varImpPlot(rf_model, main = "Random Forest Variable Importance")
dev.off()

# Save ROC Curve (Out-of-sample)
png("AnalyticsProject/results/img/class/random_forest_roc.png", width = 800, height = 600)
plot(roc(as.numeric(valid_data$homeType), rf_out_sample_probabilities), main = paste("Random Forest ROC Curve (AUC =", round(rf_out_sample_auc, 3), ")"))
dev.off()

# Append results for Random Forest
results <- rbind(results, data.frame(
  Model = "Random Forest",
  InSampleAccuracy = rf_in_sample_confusion$overall["Accuracy"],
  OutOfSampleAccuracy = rf_out_sample_confusion$overall["Accuracy"],
  AUC = rf_out_sample_auc
))

# Print and export results
print(results)
write.csv(results, "AnalyticsProject/results/classification_summary.csv", row.names = FALSE)

# Train a basic SVM model without scaling
svm_model <- svm(homeType ~ ., data = train_data, kernel = "linear", cost = 1)

# In-sample predictions
svm_in_sample_predictions <- predict(svm_model, train_data)
svm_in_sample_predictions <- factor(svm_in_sample_predictions, levels = levels(train_data$homeType))
svm_in_sample_confusion <- confusionMatrix(svm_in_sample_predictions, train_data$homeType)

# Out-of-sample predictions
svm_out_sample_predictions <- predict(svm_model, valid_data)
svm_out_sample_predictions <- factor(svm_out_sample_predictions, levels = levels(valid_data$homeType))

# Check for empty predictions
if (length(unique(svm_out_sample_predictions)) < 2) {
  stop("SVM predictions contain only one class. Check the model or data.")
}

# Out-of-sample confusion matrix
svm_out_sample_confusion <- confusionMatrix(svm_out_sample_predictions, valid_data$homeType)

# Predict probabilities for class "1" (if applicable)
svm_in_sample_probabilities <- attr(predict(svm_model, train_data, decision.values = TRUE), "decision.values")
svm_out_sample_probabilities <- attr(predict(svm_model, valid_data, decision.values = TRUE), "decision.values")

# Calculate AUC if probabilities are available
svm_in_sample_auc <- if (!is.null(svm_in_sample_probabilities)) {
  roc(as.numeric(train_data$homeType), svm_in_sample_probabilities)$auc
} else {
  NA
}

svm_out_sample_auc <- if (!is.null(svm_out_sample_probabilities)) {
  roc(as.numeric(valid_data$homeType), svm_out_sample_probabilities)$auc
} else {
  NA
}

# Save SVM ROC Curve (Out-of-sample)
if (!is.null(svm_out_sample_probabilities)) {
  png("AnalyticsProject/results/img/class/svm_roc.png", width = 800, height = 600)
  plot(roc(as.numeric(valid_data$homeType), svm_out_sample_probabilities), main = paste("SVM ROC Curve (AUC =", round(svm_out_sample_auc, 3), ")"))
  dev.off()
} else {
  message("SVM probabilities are not available for ROC curve.")
}

# Append results for SVM
results <- rbind(results, data.frame(
  Model = "SVM",
  InSampleAccuracy = svm_in_sample_confusion$overall["Accuracy"],
  OutOfSampleAccuracy = svm_out_sample_confusion$overall["Accuracy"],
  AUC = svm_out_sample_auc
))

# Save the final results table
write.csv(results, "AnalyticsProject/results/classification_results.csv", row.names = FALSE)

# Print results
print(results)


# Combine ROC curves for all models
roc_logit <- roc(valid_data$homeType, logit_out_sample_predictions)
roc_probit <- roc(valid_data$homeType, probit_out_sample_predictions)
roc_rf <- roc(as.numeric(valid_data$homeType), rf_out_sample_probabilities)
roc_svm <- if (!is.null(svm_out_sample_probabilities)) roc(as.numeric(valid_data$homeType), svm_out_sample_probabilities) else NULL

# Plot combined ROC curves
png("AnalyticsProject/results/img/class/combined_roc.png", width = 800, height = 600)
plot(roc_logit, col = "blue", lwd = 2, main = "Combined ROC Curves")
lines(roc_probit, col = "red", lwd = 2)
lines(roc_rf, col = "green", lwd = 2)
if (!is.null(roc_svm)) lines(roc_svm, col = "purple", lwd = 2)

# Add legend
legend("bottomright", legend = c(
  paste("Logistic Regression (AUC =", round(roc_logit$auc, 3), ")"),
  paste("Probit Model (AUC =", round(roc_probit$auc, 3), ")"),
  paste("Random Forest (AUC =", round(roc_rf$auc, 3), ")"),
  if (!is.null(roc_svm)) paste("SVM (AUC =", round(roc_svm$auc, 3), ")") else NULL
), col = c("blue", "red", "green", if (!is.null(roc_svm)) "purple" else NULL), lwd = 2)

dev.off()
