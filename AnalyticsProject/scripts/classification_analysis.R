# Load necessary libraries
library(tidyverse)
library(caret)
library(pROC)
library(e1071)
library(rpart)
library(randomForest)

# Load the dataset
train <- read.csv("../data/train.csv")
test <- read.csv("../data/test.csv")

# Ensure the target variable is a factor for classification
train$homeType <- as.factor(ifelse(train$homeType_Single.Family == 1, "Single Family", "Other"))
test$homeType <- as.factor(ifelse(test$homeType_Single.Family == 1, "Single Family", "Other"))

# Partition the training data into training and validation sets
set.seed(123)
train_index <- createDataPartition(train$homeType, p = 0.7, list = FALSE)
train_data <- train[train_index, ]
valid_data <- train[-train_index, ]

# Define a function to calculate performance metrics
evaluate_model <- function(model, data, target_col) {
  predictions <- predict(model, data, type = "class")
  confusion <- confusionMatrix(predictions, data[[target_col]])
  auc <- roc(as.numeric(data[[target_col]]), as.numeric(predictions))$auc
  list(confusion = confusion, auc = auc)
}

# Logistic Regression
logit_model <- glm(homeType ~ ., data = train_data, family = binomial)
logit_predictions <- predict(logit_model, valid_data, type = "response")
logit_class <- ifelse(logit_predictions > 0.5, "Single Family", "Other")
logit_confusion <- confusionMatrix(as.factor(logit_class), valid_data$homeType)
logit_auc <- roc(as.numeric(valid_data$homeType), as.numeric(logit_class))$auc

# Probit Model
probit_model <- glm(homeType ~ ., data = train_data, family = binomial(link = "probit"))
probit_predictions <- predict(probit_model, valid_data, type = "response")
probit_class <- ifelse(probit_predictions > 0.5, "Single Family", "Other")
probit_confusion <- confusionMatrix(as.factor(probit_class), valid_data$homeType)
probit_auc <- roc(as.numeric(valid_data$homeType), as.numeric(probit_class))$auc

# Decision Tree
tree_model <- rpart(homeType ~ ., data = train_data, method = "class")
tree_predictions <- predict(tree_model, valid_data, type = "class")
tree_confusion <- confusionMatrix(tree_predictions, valid_data$homeType)

# Random Forest
rf_model <- randomForest(homeType ~ ., data = train_data, ntree = 100)
rf_predictions <- predict(rf_model, valid_data)
rf_confusion <- confusionMatrix(rf_predictions, valid_data$homeType)

# Support Vector Machine
svm_model <- svm(homeType ~ ., data = train_data, kernel = "linear", probability = TRUE)
svm_predictions <- predict(svm_model, valid_data)
svm_confusion <- confusionMatrix(svm_predictions, valid_data$homeType)

# Summarize Results
results <- data.frame(
  Model = c("Logistic Regression", "Probit Model", "Decision Tree", "Random Forest", "SVM"),
  Accuracy = c(logit_confusion$overall["Accuracy"], probit_confusion$overall["Accuracy"],
               tree_confusion$overall["Accuracy"], rf_confusion$overall["Accuracy"],
               svm_confusion$overall["Accuracy"]),
  AUC = c(logit_auc, probit_auc, NA, NA, NA)  # AUC for tree and RF not calculated here
)

# Save results
write.csv(results, "../results/classification_results.csv")

# Plot the decision tree
plot(tree_model)
text(tree_model, use.n = TRUE, all = TRUE, cex = 0.8)

# Print results
print(results)