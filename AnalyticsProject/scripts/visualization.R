# Load necessary libraries
library(tidyverse)
library(ggplot2)
library(caret)
library(mgcv)
library(rpart.plot)
library(corrplot)

# Load the datasets
train <- read.csv("AnalyticsProject/data/train.csv")
test <- read.csv("AnalyticsProject/data/test.csv")

# Ensure the target variable is numeric for regression tasks
train$latestPrice <- as.numeric(train$latestPrice)
test$latestPrice <- as.numeric(test$latestPrice)

# 1. Correlation Matrix for Numeric Variables
numeric_vars <- train %>% select_if(is.numeric)
cor_matrix <- cor(numeric_vars, use = "complete.obs")

# Save the plot as a big PNG inside your project
png("AnalyticsProject/results/img/corrplot_big.png", width = 3000, height = 3000, res = 200)

corrplot::corrplot(
     cor_matrix,
     method = "color",
     type = "upper",
     tl.col = "black",
     tl.srt = 45,
     addCoef.col = "black",
     number.cex = 0.5,   # Smaller numbers
     mar = c(3, 3, 3, 3) # Bigger margins
)

dev.off()  # Close the device and save the file

# 2. Scatter Plot for Bivariate Regression
png("AnalyticsProject/results/img/scatterplot_regression.png", width = 1200, height = 800, res = 150)

# Fit the linear model
lm_model <- lm(latestPrice ~ livingAreaSqFt, data = train)
r_squared <- summary(lm_model)$r.squared

# Create the plot
ggplot(train, aes(x = livingAreaSqFt, y = latestPrice)) +
     geom_point(alpha = 0.6) +
     geom_smooth(method = "lm", color = "blue", se = FALSE) +
     annotate("text", x = max(train$livingAreaSqFt, na.rm = TRUE) * 0.8, 
              y = max(train$latestPrice, na.rm = TRUE) * 0.9, 
              label = paste("R² =", round(r_squared, 3)), 
              size = 5, color = "red") +
     labs(title = "Scatter Plot with Linear Regression Line",
                x = "Living Area (SqFt)",
                y = "Latest Price") +
     theme_minimal()

dev.off()

dev.off()  # Close the device and save the file

# 3. Polynomial Regression Visualization
png("AnalyticsProject/results/img/polynomial_regression.png", width = 1200, height = 800, res = 150)

poly_model <- lm(latestPrice ~ poly(livingAreaSqFt, 2), data = train)
train$poly_predictions <- predict(poly_model, train)
poly_r_squared <- summary(poly_model)$r.squared

ggplot(train, aes(x = livingAreaSqFt, y = latestPrice)) +
     geom_point(alpha = 0.6) +
     geom_line(aes(y = poly_predictions), color = "red") +
     annotate("text", x = max(train$livingAreaSqFt, na.rm = TRUE) * 0.8, 
              y = max(train$latestPrice, na.rm = TRUE) * 0.9, 
              label = paste("R² =", round(poly_r_squared, 3)), 
              size = 5, color = "blue") +
     labs(title = "Polynomial Regression (Degree 2)",
                x = "Living Area (SqFt)",
                y = "Latest Price") +
     theme_minimal()

dev.off()  # Close the device and save the file

# 4. GAM Visualization
png("AnalyticsProject/results/img/gam_visualization.png", width = 1200, height = 800, res = 150)

gam_model <- gam(latestPrice ~ s(livingAreaSqFt), data = train)
train$gam_predictions <- predict(gam_model, train)

# Calculate R-squared for GAM
gam_r_squared <- 1 - sum((train$latestPrice - train$gam_predictions)^2) / sum((train$latestPrice - mean(train$latestPrice))^2)

ggplot(train, aes(x = livingAreaSqFt, y = latestPrice)) +
     geom_point(alpha = 0.6) +
     geom_line(aes(y = gam_predictions), color = "green") +
     annotate("text", x = max(train$livingAreaSqFt, na.rm = TRUE) * 0.8, 
              y = max(train$latestPrice, na.rm = TRUE) * 0.9, 
              label = paste("R² =", round(gam_r_squared, 3)), 
              size = 5, color = "blue") +
     labs(title = "Generalized Additive Model (GAM)",
                x = "Living Area (SqFt)",
                y = "Latest Price") +
     theme_minimal()

dev.off()  # Close the device and save the file

# 5. Decision Tree Visualization for Classification
# Combine dummy variables into a single factor
train$homeType <- case_when(
  train$homeType_Single.Family == 1 ~ "Single Family",
  train$homeType_Condo == 1 ~ "Condo",
  train$homeType_Townhouse == 1 ~ "Townhouse",
  train$homeType_Vacant.Land == 1 ~ "Vacant Land",
  TRUE ~ "Other"
)

test$homeType <- case_when(
  test$homeType_Single.Family == 1 ~ "Single Family",
  test$homeType_Condo == 1 ~ "Condo",
  test$homeType_Townhouse == 1 ~ "Townhouse",
  test$homeType_Vacant.Land == 1 ~ "Vacant Land",
  TRUE ~ "Other"
)

# Convert to factor
train$homeType <- as.factor(train$homeType)
test$homeType <- as.factor(test$homeType)
png("AnalyticsProject/results/img/decision_tree.png", width = 1200, height = 800, res = 150)
# Decision Tree
tree_model <- rpart(homeType ~ livingAreaSqFt + yearBuilt + numOfBathrooms + numOfBedrooms, 
                    data = train, method = "class")
rpart.plot(tree_model, main = "Decision Tree for Home Type Classification")
dev.off()

# 6. Boxplot for Categorical Variables
png("AnalyticsProject/results/img/boxplot_homeType.png", width = 1200, height = 800, res = 150)
ggplot(train, aes(x = homeType, y = latestPrice, fill = homeType)) +
     geom_boxplot() +
     labs(title = "Boxplot of Latest Price by Home Type",
                x = "Home Type",
                y = "Latest Price") +
     theme_minimal()
dev.off()

# 7. ROC Curve for Classification Models
png("AnalyticsProject/results/img/roc_curve.png", width = 1200, height = 800, res = 150)
library(pROC)
logit_model <- glm(homeType_Single.Family ~ livingAreaSqFt + yearBuilt + numOfBathrooms + numOfBedrooms, data = train, family = binomial)
logit_predictions <- predict(logit_model, test, type = "response")
roc_curve <- roc(test$homeType_Single.Family, logit_predictions)

# Calculate AUC (Area Under the Curve)
auc_value <- auc(roc_curve)

# Plot the ROC curve
plot(roc_curve, main = paste("ROC Curve for Logistic Regression (AUC =", round(auc_value, 3), ")"))
dev.off()

# 8. Feature Importance for Random Forest
png("AnalyticsProject/results/img/feature_importance_rf.png", width = 1200, height = 800, res = 150)
library(randomForest)
rf_model <- randomForest(latestPrice ~ ., data = train, importance = TRUE)
varImpPlot(rf_model, main = "Feature Importance (Random Forest)")
dev.off()

# 9. Scatter Plot Matrix for Exploratory Data Analysis
png("AnalyticsProject/results/img/scatterplot_matrix.png", width = 1200, height = 800, res = 150)
library(GGally)
ggpairs(train %>% select(latestPrice, livingAreaSqFt, yearBuilt, numOfBathrooms, numOfBedrooms),
                    title = "Scatter Plot Matrix for Key Variables")
dev.off()

# 10. Histogram of Target Variable
png("AnalyticsProject/results/img/histogram_latestPrice.png", width = 1200, height = 800, res = 150)
ggplot(train, aes(x = latestPrice)) +
     geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
     labs(title = "Histogram of Latest Price",
                x = "Latest Price",
                y = "Frequency") +
     theme_minimal()
dev.off()