set.seed(42)  # Sets a fixed random seed so random operations produce the same results every time the script runs

# install.packages(c(  # Installs required packages from CRAN (only needed once per computer)
#   "tidyverse","janitor","tidymodels","textrecipes","stringr","vip",
#   "themis","ggplot2","scales","xgboost","glmnet","stopwords"
# ))

library(tidyverse)     # Loads data manipulation, visualization, and data import packages such as dplyr, ggplot2, readr, tibble
library(janitor)       # Loads functions for cleaning data and standardizing column names
library(tidymodels)    # Loads machine learning framework for preprocessing, training, tuning, and evaluation
library(textrecipes)   # Loads text preprocessing steps such as tokenization, stopword removal, and TF-IDF
library(stringr)       # Loads string manipulation functions used for cleaning text
library(vip)           # Loads functions for variable importance plots
library(themis)        # Loads techniques for handling imbalanced datasets

tidymodels::tidymodels_prefer()  # Prioritizes tidymodels functions when multiple packages contain functions with the same name

# -----------------------------
# 1) Load Data
# -----------------------------

setwd("C:/rETR")  # Sets the working directory where R will look for files such as Tweets.csv

DATA_PATH <- "Tweets.csv"  # Stores the dataset filename in a variable for easier reuse

raw <- readr::read_csv(DATA_PATH, show_col_types = FALSE) %>%  # Reads Tweets.csv into a tibble dataframe and hides column type messages
  janitor::clean_names()  # Converts column names to lowercase snake_case format for consistency

df <- raw %>%  # Starts the data cleaning pipeline
  
  transmute(  # Creates a new dataframe containing only the specified columns below
    
    text = as.character(text),  # Converts tweet text to character datatype
    
    sentiment = str_to_lower(as.character(airline_sentiment)),  # Converts sentiment labels to lowercase text
    
    confidence = airline_sentiment_confidence,  # Keeps sentiment confidence scores assigned by annotators
    
    airline = as.factor(airline),  # Converts airline names into categorical factor values
    
    created = tweet_created,  # Keeps tweet creation timestamp
    
    retweet_count = retweet_count  # Keeps number of retweets for each tweet
  ) %>%
  
  filter(!is.na(text), text != "") %>%  # Removes rows where tweet text is missing or empty
  
  filter(sentiment %in% c("negative", "neutral", "positive")) %>%  # Keeps only the three sentiment classes used for classification
  
  mutate(
    sentiment = factor(
      sentiment,
      levels = c("negative", "neutral", "positive")
    )
  )  # Converts sentiment into a factor so classification algorithms recognize it as a categorical target variable

dir.create("outputs", showWarnings = FALSE, recursive = TRUE)  # Creates an output folder if it does not already exist

# EDA plots (optional)

p_class <- df %>%  # Starts creation of sentiment class distribution plot
  
  count(sentiment) %>%  # Counts how many tweets belong to each sentiment category
  
  ggplot(aes(sentiment, n, fill = sentiment)) +  # Creates a bar chart with sentiment on x-axis and count on y-axis
  
  geom_col(show.legend = FALSE) +  # Draws bars using the count values and hides legend
  
  theme_minimal() +  # Applies a clean minimal visual style
  
  labs(
    title = "Sentiment Class Distribution",
    x = "Sentiment",
    y = "Count"
  )  # Adds chart title and axis labels

ggsave(
  "outputs/eda_class_distribution.png",
  p_class,
  width = 7,
  height = 4,
  dpi = 200
)  # Saves the sentiment distribution chart as a PNG image

p_by_airline <- df %>%  # Starts creation of sentiment distribution by airline plot
  
  count(airline, sentiment) %>%  # Counts tweets for every airline and sentiment combination
  
  group_by(airline) %>%  # Groups rows by airline so calculations occur within each airline
  
  mutate(pct = n / sum(n)) %>%  # Calculates percentage share of each sentiment within an airline
  
  ungroup() %>%  # Removes grouping structure after percentage calculation
  
  ggplot(aes(airline, pct, fill = sentiment)) +  # Creates stacked bar chart using percentage values
  
  geom_col() +  # Draws stacked bars
  
  scale_y_continuous(labels = scales::percent_format()) +  # Converts decimal values into percentages on the y-axis
  
  theme_minimal() +  # Applies minimal visual theme
  
  labs(
    title = "Sentiment Share by Airline",
    x = "Airline",
    y = "Share"
  )  # Adds chart title and axis labels

ggsave(
  "outputs/eda_sentiment_by_airline.png",
  p_by_airline,
  width = 9,
  height = 4.5,
  dpi = 200
)  # Saves the airline sentiment chart as a PNG image

# -----------------------------
# 2) Train/Test Split
# -----------------------------

set.seed(42)  # Resets random seed to ensure reproducible dataset splitting

split <- initial_split(
  df %>% select(text, sentiment),  # Keeps only predictor (text) and target (sentiment) columns
  prop = 0.80,                     # Allocates 80% of rows to training data
  strata = sentiment               # Preserves class distribution across training and testing sets
)

train <- training(split)  # Extracts the training dataset used for model fitting and tuning

test <- testing(split)    # Extracts the testing dataset used for final model evaluation

# -----------------------------
# 3) Recipe: Clean + TF-IDF
# -----------------------------

rec <- recipe(sentiment ~ text, data = train) %>%  # Creates a preprocessing recipe where sentiment is the target variable and text is the predictor
  
  step_mutate(text = str_to_lower(text)) %>%  # Converts all text to lowercase so "Flight" and "flight" are treated as the same word
  
  step_mutate(text = str_replace_all(text, "http\\S+|www\\S+", " ")) %>%  # Removes URLs such as https://example.com because links usually do not help sentiment prediction
  
  step_mutate(text = str_replace_all(text, "@\\w+", " ")) %>%  # Removes Twitter usernames such as @AmericanAir because usernames are usually not useful predictors
  
  step_mutate(text = str_replace_all(text, "#", " ")) %>%  # Removes hashtag symbols while keeping the hashtag word itself
  
  step_mutate(text = str_replace_all(text, "[^a-z\\s']", " ")) %>%  # Removes numbers, punctuation, emojis, and special characters while keeping letters, spaces, and apostrophes
  
  step_mutate(text = str_squish(text)) %>%  # Removes extra spaces and converts multiple spaces into a single space
  
  step_tokenize(text) %>%  # Splits each tweet into individual words (tokens)
  # Example:
  # "flight delayed again"
  # becomes
  # c("flight", "delayed", "again")
  
  step_stopwords(text, language = "en") %>%  # Removes common English words such as "the", "and", "is", "of"
  # because they appear frequently but carry little sentiment information
  
  step_tokenfilter(text, min_times = 5) %>%  # Removes words appearing fewer than 5 times in the training dataset
  # reduces noise and decreases dimensionality
  
  step_tfidf(text)  # Converts tokens into TF-IDF numerical features that machine learning models can use

# ----------------------------------------------------
# Example of TF-IDF
#
# TF (Term Frequency)
# Measures how often a word appears in a tweet
#
# IDF (Inverse Document Frequency)
# Gives lower importance to words appearing in many tweets
#
# Example:
#
# Word         TF-IDF Weight
# delayed      High
# cancelled    High
# flight       Low
#
# Output:
#
# tfidf_text_delayed
# tfidf_text_cancelled
# tfidf_text_customer
#
# Each word becomes a numeric predictor column.
# ----------------------------------------------------

# Prep once to estimate predictor count (for mtry range)

prep_rec <- prep(rec)  # Trains the recipe using training data and learns vocabulary, TF-IDF weights, and preprocessing rules

num_predictors <- ncol(juice(prep_rec)) - 1  # Counts the number of generated predictor columns
# subtracts 1 because the sentiment target column is not a predictor

cat("Approx TF-IDF predictors:", num_predictors, "\n")  # Prints the approximate number of TF-IDF features created

# -----------------------------
# 4) Models
# -----------------------------

model_glmnet <- multinom_reg(
  penalty = tune(),  # Regularization strength will be optimized during hyperparameter tuning
  mixture = tune()   # Controls Lasso vs Ridge regularization and will also be tuned
) %>%
  
  set_engine("glmnet") %>%  # Uses the glmnet package implementation
  
  set_mode("classification")  # Specifies multiclass classification task

# ----------------------------------------------------
# Multinomial Logistic Regression
#
# Baseline model used for comparison.
#
# Advantages:
# - Fast
# - Interpretable
# - Strong baseline for text classification
#
# penalty:
# Controls overfitting
#
# mixture:
# 0 = Ridge
# 1 = Lasso
# 0.5 = Elastic Net
# ----------------------------------------------------

model_xgb <- boost_tree(
  
  trees = tune(),  # Number of boosting trees to build
  
  tree_depth = tune(),  # Maximum depth of each tree
  # Larger values can capture more complex patterns
  
  learn_rate = tune(),  # Controls how much each tree contributes
  # Smaller values often improve generalization
  
  mtry = tune(),  # Number of predictors randomly sampled at each split
  
  min_n = tune(),  # Minimum number of observations required to split a node
  
  loss_reduction = tune(),  # Minimum improvement required before making a split
  # Helps prevent unnecessary tree growth
  
  sample_size = tune()  # Fraction of training data randomly sampled for each tree
  # Helps reduce overfitting
) %>%
  
  set_engine(
    "xgboost",
    
    objective = "multi:softprob",  # Produces class probabilities for multiclass classification
    
    eval_metric = "mlogloss"       # Uses multiclass log loss as evaluation metric
  ) %>%
  
  set_mode("classification")  # Specifies classification task

# ----------------------------------------------------
# XGBoost
#
# Extreme Gradient Boosting
#
# Ensemble learning algorithm.
#
# Builds many decision trees sequentially.
#
# Each new tree attempts to correct errors
# made by previous trees.
#
# Advantages:
# - Usually higher accuracy
# - Handles high-dimensional TF-IDF data well
# - Captures non-linear relationships
# ----------------------------------------------------

wf_glmnet <- workflow() %>%  # Creates a machine learning workflow object
  add_recipe(rec) %>%        # Adds preprocessing recipe
  add_model(model_glmnet)    # Adds logistic regression model

wf_xgb <- workflow() %>%     # Creates another workflow object
  add_recipe(rec) %>%        # Uses the same preprocessing pipeline
  add_model(model_xgb)       # Adds XGBoost model

# ----------------------------------------------------
# Workflow
#
# Combines:
# 1. Preprocessing steps
# 2. Machine learning model
#
# Benefits:
# - Prevents data leakage
# - Simplifies tuning
# - Ensures identical preprocessing
#   across train/test data
# ----------------------------------------------------

# -----------------------------
# 5) CV + Metrics for tuning (compat)
# -----------------------------

set.seed(42)  # Ensures cross-validation folds are reproducible

folds <- vfold_cv(
  train,
  v = 5,                 # Creates 5-fold cross-validation
  strata = sentiment     # Preserves class distribution in each fold
)

# ----------------------------------------------------
# 5-Fold Cross Validation
#
# Training data is divided into 5 parts.
#
# Iteration 1:
# Train on 4 folds
# Validate on 1 fold
#
# Iteration 2:
# Train on different 4 folds
# Validate on remaining fold
#
# Repeated until every fold has been used
# as validation exactly once.
#
# Final score = average across all folds.
# ----------------------------------------------------

# Use metrics that are always compatible:
# - accuracy: class metric
# - mn_log_loss: class probability metric (works for multiclass)

metric_set_tune <- metric_set(
  accuracy,      # Measures proportion of correctly classified tweets
  mn_log_loss    # Measures quality of predicted probabilities
)