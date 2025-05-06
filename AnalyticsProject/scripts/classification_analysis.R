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
  Accuracy = numeric(),
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
logit_predictions <- predict(logit_model, valid_data, type = "response")
logit_class <- ifelse(logit_predictions > 0.5, 1, 0)
logit_confusion <- confusionMatrix(as.factor(logit_class), as.factor(valid_data$homeType))

# Calculate AUC using probabilities
logit_auc <- roc(valid_data$homeType, logit_predictions)$auc

# Save Logistic Regression ROC Curve
png("AnalyticsProject/results/img/class/logistic_regression_roc.png", width = 800, height = 600)
plot(roc(valid_data$homeType, logit_predictions), main = paste("Logistic Regression ROC Curve (AUC =", round(logit_auc, 3), ")"))
dev.off()

# Append results for Logistic Regression
results <- rbind(results, data.frame(
  Model = "Logistic Regression",
  Accuracy = logit_confusion$overall["Accuracy"],
  AUC = logit_auc
))

# Probit Model
probit_model <- glm(homeType ~ ., data = train_data, family = binomial(link = "probit"))
probit_predictions <- predict(probit_model, valid_data, type = "response")
probit_class <- ifelse(probit_predictions > 0.5, 1, 0)
probit_confusion <- confusionMatrix(as.factor(probit_class), as.factor(valid_data$homeType))

# Calculate AUC using probabilities
probit_auc <- roc(valid_data$homeType, probit_predictions)$auc

# Save Probit Model ROC Curve
png("AnalyticsProject/results/img/class/probit_model_roc.png", width = 800, height = 600)
plot(roc(valid_data$homeType, probit_predictions), main = paste("Probit Model ROC Curve (AUC =", round(probit_auc, 3), ")"))
dev.off()

# Append results for Probit Model
results <- rbind(results, data.frame(
  Model = "Probit Model",
  Accuracy = probit_confusion$overall["Accuracy"],
  AUC = probit_auc
))

# Decision Tree
tree_model <- rpart(homeType ~ ., data = train_data, method = "class")
tree_predictions <- predict(tree_model, valid_data, type = "class")
tree_confusion <- confusionMatrix(as.factor(tree_predictions), as.factor(valid_data$homeType))

# Save Decision Tree Plot
png("AnalyticsProject/results/img/class/decision_tree.png", width = 800, height = 600)
plot(tree_model)
text(tree_model, use.n = TRUE, all = TRUE, cex = 0.8)
dev.off()

# Append results for Decision Tree
results <- rbind(results, data.frame(
  Model = "Decision Tree",
  Accuracy = tree_confusion$overall["Accuracy"],
  AUC = NA  # AUC not applicable for Decision Tree
))

# Random Forest
rf_model <- randomForest(homeType ~ ., data = train_data, ntree = 50)
rf_predictions <- predict(rf_model, valid_data)
rf_confusion <- confusionMatrix(as.factor(rf_predictions), as.factor(valid_data$homeType))

# Save Random Forest Variable Importance Plot
png("AnalyticsProject/results/img/class/random_forest_importance.png", width = 800, height = 600)
varImpPlot(rf_model, main = "Random Forest Variable Importance")
dev.off()

# Append results for Random Forest
results <- rbind(results, data.frame(
  Model = "Random Forest",
  Accuracy = rf_confusion$overall["Accuracy"],
  AUC = NA  # AUC not calculated for Random Forest
))

# Oversample the minority class
oversample <- function(data, target_col) {
  # Split the data into majority and minority classes
  majority <- data[data[[target_col]] == 0, ]
  minority <- data[data[[target_col]] == 1, ]
  
  # Oversample the minority class
  oversampled_minority <- minority[sample(1:nrow(minority), size = nrow(majority), replace = TRUE), ]
  
  # Combine the majority and oversampled minority classes
  balanced_data <- rbind(majority, oversampled_minority)
  return(balanced_data)
}

# Apply oversampling to the training data
train_data <- oversample(train_data, "homeType")

# Scale the data
train_data_scaled <- train_data %>% mutate(across(-homeType, scale))
valid_data_scaled <- valid_data %>% mutate(across(-homeType, scale))

# Train SVM with tuned hyperparameters
svm_tuned <- tune(svm, homeType ~ ., data = train_data_scaled, 
                  kernel = "linear", 
                  ranges = list(cost = c(0.1, 1, 10, 100)))
svm_model <- svm_tuned$best.model

# Predict on validation data
svm_predictions <- predict(svm_model, valid_data_scaled)

# Ensure levels match between predictions and reference
svm_predictions <- factor(svm_predictions, levels = levels(valid_data$homeType))

# Check for empty predictions
if (length(unique(svm_predictions)) < 2) {
  stop("SVM predictions contain only one class. Check the model or data.")
}

# Calculate confusion matrix
svm_confusion <- confusionMatrix(svm_predictions, as.factor(valid_data$homeType))

# Append results for SVM
results <- rbind(results, data.frame(
  Model = "SVM",
  Accuracy = svm_confusion$overall["Accuracy"],
  AUC = NA  # AUC not calculated for SVM
))

# Save the final results table
write.csv(results, "AnalyticsProject/results/classification_results.csv", row.names = FALSE)

# Print results
print(results)
