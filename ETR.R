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

# -----------------------------
# 6) Tuning
# -----------------------------

set.seed(42)  # Sets random seed so hyperparameter tuning results are reproducible

grid_glmnet <- grid_regular(  # Creates a regular grid of hyperparameter combinations for glmnet
  
  penalty(range = c(-4, 0)),  # Creates penalty values on a log10 scale
  # -4 = 10^-4 = 0.0001
  #  0 = 10^0 = 1
  # Controls regularization strength
  
  mixture(),                  # Creates values between 0 and 1
  # 0 = Ridge Regression
  # 1 = Lasso Regression
  # Values in between = Elastic Net
  
  levels = 6                  # Generates 6 values for each parameter
  # Produces a grid of combinations to test
)

tuned_glmnet <- tune_grid(    # Performs hyperparameter tuning using cross-validation
  
  wf_glmnet,                  # Workflow containing recipe + glmnet model
  
  resamples = folds,          # Uses the previously created 5-fold cross-validation folds
  
  grid = grid_glmnet,         # Tests every parameter combination in the grid
  
  metrics = metric_set_tune   # Evaluates each combination using accuracy and log loss
)

# ----------------------------------------------------
# mtry controls how many predictor variables are
# randomly considered at each split in XGBoost.
#
# Since TF-IDF can create thousands of predictors,
# we calculate a reasonable search range dynamically.
# ----------------------------------------------------

mtry_low <- max(
  10L,                        # Minimum lower bound of 10 predictors
  floor(num_predictors * 0.02) # Or 2% of all predictors
)

mtry_high <- max(
  mtry_low + 10L,             # Ensures upper bound is larger than lower bound
  floor(num_predictors * 0.15) # Or 15% of all predictors
)

set.seed(42)  # Makes XGBoost tuning reproducible

grid_xgb <- grid_latin_hypercube(  # Creates an efficient random hyperparameter search space
  
  trees(range = c(300L, 1200L)),  # Tests between 300 and 1200 boosting trees
  
  tree_depth(range = c(2L, 10L)), # Tests tree depths between 2 and 10
  # Larger depth captures more complex patterns
  
  learn_rate(range = c(-4, -1)),  # Tests learning rates from 10^-4 to 10^-1
  # Lower values learn more slowly but often generalize better
  
  mtry(range = c(mtry_low, mtry_high)),  # Tests different numbers of predictors at each split
  
  min_n(range = c(2L, 40L)),  # Tests minimum observations required to create a split
  
  loss_reduction(),           # Tests minimum gain required before making a split
  # Similar to pruning
  
  sample_size = sample_prop(range = c(0.6, 1.0)),  # Tests using 60%-100% of data per tree
  
  size = 25                   # Generates 25 unique parameter combinations
)

tuned_xgb <- tune_grid(       # Performs XGBoost hyperparameter tuning
  
  wf_xgb,                     # Workflow containing recipe + XGBoost model
  
  resamples = folds,          # Uses 5-fold cross-validation
  
  grid = grid_xgb,            # Uses the generated Latin Hypercube search space
  
  metrics = metric_set_tune   # Evaluates using accuracy and multiclass log loss
)

# Saves tuning results to disk.
# Allows analysis later without rerunning expensive tuning.
saveRDS(tuned_glmnet, "outputs/tuned_glmnet.rds")

saveRDS(tuned_xgb, "outputs/tuned_xgb.rds")

# ----------------------------------------------------
# select_best()
#
# Chooses the hyperparameter combination that
# achieved the best score.
#
# metric = "mn_log_loss"
#
# Lower log loss means:
# - better probability estimates
# - higher confidence in correct predictions
# ----------------------------------------------------

best_glmnet <- select_best(
  tuned_glmnet,
  metric = "mn_log_loss"
)

best_xgb <- select_best(
  tuned_xgb,
  metric = "mn_log_loss"
)

# ----------------------------------------------------
# finalize_workflow()
#
# Inserts the best hyperparameter values
# into the workflow.
#
# fit()
#
# Trains the final model on the entire
# training dataset.
# ----------------------------------------------------

final_glmnet <- finalize_workflow(
  wf_glmnet,
  best_glmnet
) %>%
  fit(data = train)

final_xgb <- finalize_workflow(
  wf_xgb,
  best_xgb
) %>%
  fit(data = train)

# -----------------------------
# 7) Test Evaluation
# -----------------------------

eval_test <- function(fitted_wf, test_df) {  # Creates a reusable evaluation function
  
  preds_prob <- predict(
    fitted_wf,
    test_df,
    type = "prob"
  )  # Generates predicted probabilities for each class
  
  preds_cls <- predict(
    fitted_wf,
    test_df,
    type = "class"
  )  # Generates final predicted class labels
  
  preds <- bind_cols(
    test_df,
    preds_prob,
    preds_cls
  )  # Combines original data, probabilities, and predictions into one table
  
  acc <- yardstick::accuracy(
    preds,
    truth = sentiment,
    estimate = .pred_class
  )  # Calculates classification accuracy
  
  macro_f1 <- yardstick::f_meas(
    preds,
    truth = sentiment,
    estimate = .pred_class,
    estimator = "macro"
  )  # Calculates Macro F1 score
  
  # Macro F1:
  # Calculates F1 for each class separately
  # Then averages them equally
  #
  # Useful when class distribution is imbalanced
  
  cm <- yardstick::conf_mat(
    preds,
    truth = sentiment,
    estimate = .pred_class
  )  # Creates confusion matrix
  
  list(
    accuracy = acc,
    macro_f1 = macro_f1,
    conf_mat = cm,
    preds = preds
  )  # Returns all evaluation results as a list
}

# Evaluates final logistic regression model
res_glmnet <- eval_test(
  final_glmnet,
  test
)

# Evaluates final XGBoost model
res_xgb <- eval_test(
  final_xgb,
  test
)

# Prints test results header
cat("\n--- Test results ---\n")

# Displays logistic regression performance
cat("\nGLMNET:\n")
print(res_glmnet$accuracy)
print(res_glmnet$macro_f1)

# Displays XGBoost performance
cat("\nXGBoost:\n")
print(res_xgb$accuracy)
print(res_xgb$macro_f1)

# ----------------------------------------------------
# Creates a confusion matrix heatmap.
#
# Confusion Matrix:
#
# Actual vs Predicted classes.
#
# Helps identify:
# - correctly classified examples
# - common classification mistakes
# ----------------------------------------------------

conf_plot <- autoplot(
  res_xgb$conf_mat,
  type = "heatmap"
) +
  ggtitle("Confusion Matrix (XGBoost) - Test Set") +
  theme_minimal()

ggsave(
  "outputs/confusion_matrix_xgb.png",
  conf_plot,
  width = 7,
  height = 5,
  dpi = 200
)  # Saves confusion matrix visualization

# Creates summary table containing final model performance

metrics_summary <- tibble(
  
  model = c(
    "glmnet_multinom",
    "xgboost"
  ),  # Model names
  
  accuracy = c(
    as.numeric(res_glmnet$accuracy$.estimate),
    as.numeric(res_xgb$accuracy$.estimate)
  ),  # Converts accuracy values to numeric format
  
  macro_f1 = c(
    as.numeric(res_glmnet$macro_f1$.estimate),
    as.numeric(res_xgb$macro_f1$.estimate)
  )  # Converts Macro F1 values to numeric format
)

readr::write_csv(
  metrics_summary,
  "outputs/test_metrics_summary.csv"
)  # Saves performance summary table as CSV file

# -----------------------------
# 8) Interpretation (XGBoost feature importance)
# -----------------------------

try({  # Runs the code inside the block and prevents the script from stopping if an error occurs
  
  fit_xgb <- extract_fit_parsnip(final_xgb)$fit  # Extracts the underlying trained XGBoost model from the tidymodels workflow
  
  imp <- xgboost::xgb.importance(model = fit_xgb)  # Calculates feature importance scores for all TF-IDF predictors
  
  readr::write_csv(
    as_tibble(imp),
    "outputs/xgb_feature_importance.csv"
  )  # Saves the complete feature importance table as a CSV file
  
  top_imp <- imp %>%
    as_tibble() %>%  # Converts importance output into a tibble
    
    slice_max(
      order_by = Gain,
      n = 30
    )  # Selects the 30 most important features based on Gain
  
  # ----------------------------------------------------
  # Gain
  #
  # Measures how much a feature contributes
  # to reducing prediction error.
  #
  # Higher Gain = More important feature
  #
  # Example:
  #
  # Feature      Gain
  # delayed      0.082
  # cancelled    0.065
  # thanks       0.053
  #
  # "delayed" contributed more to predictions
  # than "thanks".
  # ----------------------------------------------------
  
  p_imp <- top_imp %>%
    
    ggplot(
      aes(
        x = reorder(Feature, Gain),  # Orders features from lowest to highest gain
        y = Gain                     # Uses gain values as bar heights
      )
    ) +
    
    geom_col() +  # Creates bar chart
    
    coord_flip() +  # Flips axes to make feature names easier to read
    
    theme_minimal() +  # Applies clean visual theme
    
    labs(
      title = "Top 30 TF-IDF Features (XGBoost Importance)",
      x = "Token",
      y = "Gain"
    )  # Adds chart title and axis labels
  
  ggsave(
    "outputs/xgb_top_features.png",
    p_imp,
    width = 8,
    height = 8,
    dpi = 200
  )  # Saves feature importance visualization
  
}, silent = TRUE)  # Hides error messages if feature importance extraction fails

# ----------------------------------------------------
# Why use feature importance?
#
# It helps explain:
# - Which words influenced predictions most
# - Why the model made decisions
# - Which TF-IDF features were most useful
#
# This improves model interpretability.
# ----------------------------------------------------

# -----------------------------
# 9) Demo predictions
# -----------------------------

demo_texts <- tibble(  # Creates a small dataframe containing example tweets
  
  text = c(
    
    "Flight delayed again and the staff was rude. Terrible experience.",
    
    "Thanks for the quick help at the gate, really appreciate it!",
    
    "My flight is scheduled as usual. No updates yet."
    
  )
)

# ----------------------------------------------------
# Expected sentiment:
#
# Tweet 1 → Negative
# Tweet 2 → Positive
# Tweet 3 → Neutral
#
# Used to demonstrate how the model predicts
# unseen text data.
# ----------------------------------------------------

demo_pred <- demo_texts %>%
  
  bind_cols(
    predict(
      final_xgb,
      demo_texts,
      type = "prob"
    )
  ) %>%  # Adds predicted probabilities for each sentiment class
  
  bind_cols(
    predict(
      final_xgb,
      demo_texts,
      type = "class"
    )
  )  # Adds final predicted sentiment label

# Example output:
#
# text
# .pred_negative
# .pred_neutral
# .pred_positive
# .pred_class
#
# Each row contains:
# - Probability of each sentiment
# - Final predicted class

print(demo_pred)  # Displays prediction results in the console

readr::write_csv(
  demo_pred,
  "outputs/demo_predictions.csv"
)  # Saves prediction results as CSV

# -----------------------------
# Save models + session info
# -----------------------------

saveRDS(
  final_glmnet,
  "outputs/final_glmnet_workflow.rds"
)  # Saves trained logistic regression workflow for future use

saveRDS(
  final_xgb,
  "outputs/final_xgb_workflow.rds"
)  # Saves trained XGBoost workflow for future use

writeLines(
  capture.output(sessionInfo()),
  "outputs/session_info.txt"
)  # Saves R session information including package versions

# ----------------------------------------------------
# sessionInfo() contains:
#
# - R version
# - Operating system
# - Installed package versions
#
# Useful for:
# - Reproducibility
# - Debugging
# - Documentation
# ----------------------------------------------------

cat(
  "\nDONE. Outputs saved in ./outputs\n"
)  # Prints completion message indicating that all files were successfully generated