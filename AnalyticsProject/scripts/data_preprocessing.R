# Load csv and check data

library(tidyverse)
library(fastDummies)
library(rsample)

url <- "https://raw.githubusercontent.com/SantiagoArenas/Portfolio/refs/heads/main/Machine_Learning/austinHousingData.csv"
db <- read_csv(url)

glimpse(db)

# Drop irrelevant columns and drop rows with missing values
db <- db %>%
  select(-homeImage, -zpid, -streetAddress, -zipcode, -description, 
         -latest_salemonth, -latest_saleyear) %>%
  drop_na()

# Check counts of each home type and city
table(db$homeType)
table(db$city)

# keep only top 5 home types and city
top_home_types <- db %>%
  count(homeType, sort = TRUE) %>%
  top_n(5) %>%
  pull(homeType)

top_cities <- db %>%
  count(city, sort = TRUE) %>%
  top_n(5) %>%
  pull(city)

db <- db %>%
  filter(homeType %in% top_home_types,
         city %in% top_cities)

# Make dummy variables for categorical variables
db <- db %>%
  mutate(across(c(city, homeType), as.factor)) %>%
  fastDummies::dummy_cols(select_columns = c("city", "homeType"), remove_first_dummy = FALSE, remove_selected_columns = TRUE)

# Check head of data
head(db)

# Train-test split and save the csv's
set.seed(42)
split <- initial_split(db, prop = 0.8)
train <- training(split)
test <- testing(split)

write_csv(train, "train.csv")
write_csv(test, "test.csv") # Then we move it to the data folder

glimpse(train)
glimpse(test)

head(train)
head(test)