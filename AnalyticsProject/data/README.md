# Dataset Overview

This project uses two datasets: `train.csv` and `test.csv`. Both datasets contain information about real estate properties, including their features, location, and pricing. These datasets are used for predictive modeling tasks, including regression and classification.

## File Descriptions

### `train.csv`
- **Purpose**: This dataset is used for training predictive models. It contains labeled data with features and target variables.
- **Number of Rows**: 500+ (exact count depends on the dataset split).
- **Number of Columns**: 80+ (features and target variables).
- **Target Variables**:
  - `latestPrice`: The most recent price of the property (used for regression tasks).
  - `homeType_*`: Encoded categorical variables indicating the type of home (used for classification tasks).
- **Features**:
  - **Location**: `latitude`, `longitude`.
  - **Property Details**: `yearBuilt`, `garageSpaces`, `parkingSpaces`, `lotSizeSqFt`, `livingAreaSqFt`.
  - **Amenities**: `hasGarage`, `hasCooling`, `hasHeating`, `hasSpa`, `hasView`.
  - **School Information**: `avgSchoolDistance`, `avgSchoolRating`, `avgSchoolSize`.
  - **Other Features**: `numOfBathrooms`, `numOfBedrooms`, `numOfStories`, `propertyTaxRate`.

### `test.csv`
- **Purpose**: This dataset is used for testing and validating the predictive models. It contains the same features as `train.csv` but is used to evaluate model performance.
- **Number of Rows**: Similar to `train.csv` (split during preprocessing).
- **Number of Columns**: Same as `train.csv`.

## Data Preprocessing
Before using the datasets for modeling, the following preprocessing steps are applied:
1. **Handling Missing Values**: Rows or columns with missing values are either imputed or removed.
2. **Feature Engineering**: New features are created, such as aggregating school-related variables or encoding categorical variables.
3. **Normalization/Scaling**: Continuous variables are normalized to ensure consistent scaling.
4. **Train-Test Split**: The data is split into training and testing sets (if not already split).

## Predictive Tasks
1. **Regression**: Predicting the `latestPrice` of properties based on their features.
2. **Classification**: Classifying properties into different `homeType` categories.

## Notes
- The datasets are cross-sectional, meaning they represent a snapshot of property data at a specific time.
- The `latest_saledate` column is converted into a datetime format for potential time-based feature engineering.
- Categorical variables (e.g., `city_*`, `homeType_*`) are one-hot encoded for use in machine learning models.

For more details on the features and their roles in the predictive tasks, refer to the project documentation.