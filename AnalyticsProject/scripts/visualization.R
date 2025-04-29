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
png("AnalyticsProject/results/img/corrplot_big.png", width = 2000, height = 2000, res = 200)

corrplot::corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  tl.col = "black",
  tl.srt = 45,
  addCoef.col = "black",
  number.cex = 1.2,   # Bigger numbers
  mar = c(3, 3, 3, 3) # Bigger margins
)

dev.off()  # Close the device and save the file

# 2. Scatter Plot for Bivariate Regression
ggplot(train, aes(x = livingAreaSqFt, y = latestPrice)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", color = "blue", se = FALSE) +
  labs(title = "Scatter Plot with Linear Regression Line",
       x = "Living Area (SqFt)",
       y = "Latest Price") +
  theme_minimal()

# 3. Polynomial Regression Visualization
poly_model <- lm(latestPrice ~ poly(livingAreaSqFt, 2), data = train)
train$poly_predictions <- predict(poly_model, train)
ggplot(train, aes(x = livingAreaSqFt, y = latestPrice)) +
  geom_point(alpha = 0.6) +
  geom_line(aes(y = poly_predictions), color = "red") +
  labs(title = "Polynomial Regression (Degree 2)",
       x = "Living Area (SqFt)",
       y = "Latest Price") +
  theme_minimal()

# 4. GAM Visualization
gam_model <- gam(latestPrice ~ s(livingAreaSqFt), data = train)
train$gam_predictions <- predict(gam_model, train)
ggplot(train, aes(x = livingAreaSqFt, y = latestPrice)) +
  geom_point(alpha = 0.6) +
  geom_line(aes(y = gam_predictions), color = "green") +
  labs(title = "Generalized Additive Model (GAM)",
       x = "Living Area (SqFt)",
       y = "Latest Price") +
  theme_minimal()

# 5. Decision Tree Visualization for Classification
tree_model <- rpart(homeType ~ livingAreaSqFt + yearBuilt + numOfBathrooms + numOfBedrooms, data = train, method = "class")
rpart.plot(tree_model, main = "Decision Tree for Home Type Classification")

# 6. Boxplot for Categorical Variables
ggplot(train, aes(x = homeType, y = latestPrice, fill = homeType)) +
  geom_boxplot() +
  labs(title = "Boxplot of Latest Price by Home Type",
       x = "Home Type",
       y = "Latest Price") +
  theme_minimal()

# 7. ROC Curve for Classification Models
library(pROC)
logit_model <- glm(homeType_Single.Family ~ livingAreaSqFt + yearBuilt + numOfBathrooms + numOfBedrooms, data = train, family = binomial)
logit_predictions <- predict(logit_model, test, type = "response")
roc_curve <- roc(test$homeType_Single.Family, logit_predictions)
plot(roc_curve, main = "ROC Curve for Logistic Regression")

# 8. Feature Importance for Random Forest
library(randomForest)
rf_model <- randomForest(latestPrice ~ ., data = train, importance = TRUE)
varImpPlot(rf_model, main = "Feature Importance (Random Forest)")

# 9. Scatter Plot Matrix for Exploratory Data Analysis
library(GGally)
ggpairs(train %>% select(latestPrice, livingAreaSqFt, yearBuilt, numOfBathrooms, numOfBedrooms),
        title = "Scatter Plot Matrix for Key Variables")

# 10. Histogram of Target Variable
ggplot(train, aes(x = latestPrice)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  labs(title = "Histogram of Latest Price",
       x = "Latest Price",
       y = "Frequency") +
  theme_minimal()