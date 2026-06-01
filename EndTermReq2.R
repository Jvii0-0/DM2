# ============================================================
# End-term Data Mining Project (Text Mining + Ensemble ML)
# Dataset: Twitter US Airline Sentiment (Kaggle / Crowdflower)
# Task: 3-class sentiment classification (negative/neutral/positive)
# Advanced ML: Text mining (TF-IDF) + XGBoost (ensemble)
# Baseline: Multinomial Logistic Regression (glmnet)
# Decision: NO confidence filtering (use all labeled tweets)
# Date: 2026-05-17
# ============================================================

set.seed(42)

# install.packages(c(
#   "tidyverse","janitor","tidymodels","textrecipes","stringr","vip",
#   "themis","ggplot2","scales","xgboost","glmnet","stopwords"
# ))

library(tidyverse)
library(janitor)
library(tidymodels)
library(textrecipes)
library(stringr)
library(vip)
library(themis)

tidymodels::tidymodels_prefer()

# -----------------------------
# 1) Load Data
# -----------------------------
setwd("C:/rETR")
DATA_PATH <- "Tweets.csv"

raw <- readr::read_csv(DATA_PATH, show_col_types = FALSE) %>%
  janitor::clean_names()

df <- raw %>%
  transmute(
    text = as.character(text),
    sentiment = str_to_lower(as.character(airline_sentiment)),
    confidence = airline_sentiment_confidence,
    airline = as.factor(airline),
    created = tweet_created,
    retweet_count = retweet_count
  ) %>%
  filter(!is.na(text), text != "") %>%
  filter(sentiment %in% c("negative", "neutral", "positive")) %>%
  mutate(sentiment = factor(sentiment, levels = c("negative", "neutral", "positive")))

dir.create("outputs", showWarnings = FALSE, recursive = TRUE)

# EDA plots (optional)
p_class <- df %>%
  count(sentiment) %>%
  ggplot(aes(sentiment, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  theme_minimal() +
  labs(title = "Sentiment Class Distribution", x = "Sentiment", y = "Count")
ggsave("outputs/eda_class_distribution.png", p_class, width = 7, height = 4, dpi = 200)

p_by_airline <- df %>%
  count(airline, sentiment) %>%
  group_by(airline) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup() %>%
  ggplot(aes(airline, pct, fill = sentiment)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_minimal() +
  labs(title = "Sentiment Share by Airline", x = "Airline", y = "Share")
ggsave("outputs/eda_sentiment_by_airline.png", p_by_airline, width = 9, height = 4.5, dpi = 200)

# -----------------------------
# 2) Train/Test Split
# -----------------------------
set.seed(42)
split <- initial_split(df %>% select(text, sentiment), prop = 0.80, strata = sentiment)
train <- training(split)
test  <- testing(split)

# -----------------------------
# 3) Recipe: Clean + TF-IDF
# -----------------------------
rec <- recipe(sentiment ~ text, data = train) %>%
  step_mutate(text = str_to_lower(text)) %>%
  step_mutate(text = str_replace_all(text, "http\\S+|www\\S+", " ")) %>%
  step_mutate(text = str_replace_all(text, "@\\w+", " ")) %>%
  step_mutate(text = str_replace_all(text, "#", " ")) %>%
  step_mutate(text = str_replace_all(text, "[^a-z\\s']", " ")) %>%
  step_mutate(text = str_squish(text)) %>%
  step_tokenize(text) %>%
  step_stopwords(text, language = "en") %>%
  step_tokenfilter(text, min_times = 5) %>%
  step_tfidf(text)

# Prep once to estimate predictor count (for mtry range)
prep_rec <- prep(rec)
num_predictors <- ncol(juice(prep_rec)) - 1
cat("Approx TF-IDF predictors:", num_predictors, "\n")

# -----------------------------
# 4) Models
# -----------------------------
model_glmnet <- multinom_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

model_xgb <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  mtry = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune()
) %>%
  set_engine("xgboost", objective = "multi:softprob", eval_metric = "mlogloss") %>%
  set_mode("classification")

wf_glmnet <- workflow() %>% add_recipe(rec) %>% add_model(model_glmnet)
wf_xgb    <- workflow() %>% add_recipe(rec) %>% add_model(model_xgb)

# -----------------------------
# 5) CV + Metrics for tuning (compat)
# -----------------------------
set.seed(42)
folds <- vfold_cv(train, v = 5, strata = sentiment)

# Use metrics that are always compatible:
# - accuracy: class metric
# - mn_log_loss: class probability metric (works for multiclass)
metric_set_tune <- metric_set(accuracy, mn_log_loss)

# -----------------------------
# 6) Tuning
# -----------------------------
set.seed(42)
grid_glmnet <- grid_regular(
  penalty(range = c(-4, 0)),
  mixture(),
  levels = 6
)

tuned_glmnet <- tune_grid(
  wf_glmnet,
  resamples = folds,
  grid = grid_glmnet,
  metrics = metric_set_tune
)

mtry_low  <- max(10L, floor(num_predictors * 0.02))
mtry_high <- max(mtry_low + 10L, floor(num_predictors * 0.15))

set.seed(42)
grid_xgb <- grid_latin_hypercube(
  trees(range = c(300L, 1200L)),
  tree_depth(range = c(2L, 10L)),
  learn_rate(range = c(-4, -1)),
  mtry(range = c(mtry_low, mtry_high)),
  min_n(range = c(2L, 40L)),
  loss_reduction(),
  sample_size = sample_prop(range = c(0.6, 1.0)),
  size = 25
)

tuned_xgb <- tune_grid(
  wf_xgb,
  resamples = folds,
  grid = grid_xgb,
  metrics = metric_set_tune
)

saveRDS(tuned_glmnet, "outputs/tuned_glmnet.rds")
saveRDS(tuned_xgb, "outputs/tuned_xgb.rds")

# Pick best by log loss (probability quality) or accuracy
best_glmnet <- select_best(tuned_glmnet, metric = "mn_log_loss")
best_xgb    <- select_best(tuned_xgb, metric = "mn_log_loss")

final_glmnet <- finalize_workflow(wf_glmnet, best_glmnet) %>% fit(data = train)
final_xgb    <- finalize_workflow(wf_xgb, best_xgb) %>% fit(data = train)

# -----------------------------
# 7) Test Evaluation (includes Macro-F1 requirement)
# -----------------------------
eval_test <- function(fitted_wf, test_df) {
  preds_prob <- predict(fitted_wf, test_df, type = "prob")
  preds_cls  <- predict(fitted_wf, test_df, type = "class")

  preds <- bind_cols(test_df, preds_prob, preds_cls)

  acc <- yardstick::accuracy(preds, truth = sentiment, estimate = .pred_class)
  macro_f1 <- yardstick::f_meas(preds, truth = sentiment, estimate = .pred_class, estimator = "macro")
  cm <- yardstick::conf_mat(preds, truth = sentiment, estimate = .pred_class)

  list(accuracy = acc, macro_f1 = macro_f1, conf_mat = cm, preds = preds)
}

res_glmnet <- eval_test(final_glmnet, test)
res_xgb    <- eval_test(final_xgb, test)

cat("\n--- Test results ---\n")
cat("\nGLMNET:\n"); print(res_glmnet$accuracy); print(res_glmnet$macro_f1)
cat("\nXGBoost:\n"); print(res_xgb$accuracy);    print(res_xgb$macro_f1)

# Save confusion matrix plot
conf_plot <- autoplot(res_xgb$conf_mat, type = "heatmap") +
  ggtitle("Confusion Matrix (XGBoost) - Test Set") +
  theme_minimal()
ggsave("outputs/confusion_matrix_xgb.png", conf_plot, width = 7, height = 5, dpi = 200)

# Save metrics summary
metrics_summary <- tibble(
  model = c("glmnet_multinom", "xgboost"),
  accuracy = c(as.numeric(res_glmnet$accuracy$.estimate), as.numeric(res_xgb$accuracy$.estimate)),
  macro_f1  = c(as.numeric(res_glmnet$macro_f1$.estimate), as.numeric(res_xgb$macro_f1$.estimate))
)
readr::write_csv(metrics_summary, "outputs/test_metrics_summary.csv")

# -----------------------------
# 8) Interpretation (XGBoost feature importance)
# -----------------------------
try({
  fit_xgb <- extract_fit_parsnip(final_xgb)$fit
  imp <- xgboost::xgb.importance(model = fit_xgb)
  readr::write_csv(as_tibble(imp), "outputs/xgb_feature_importance.csv")

  top_imp <- imp %>% as_tibble() %>% slice_max(order_by = Gain, n = 30)
  p_imp <- top_imp %>%
    ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
    geom_col() +
    coord_flip() +
    theme_minimal() +
    labs(title = "Top 30 TF-IDF Features (XGBoost Importance)", x = "Token", y = "Gain")
  ggsave("outputs/xgb_top_features.png", p_imp, width = 8, height = 8, dpi = 200)
}, silent = TRUE)

# -----------------------------
# 9) Demo predictions
# -----------------------------
demo_texts <- tibble(text = c(
  "Flight delayed again and the staff was rude. Terrible experience.",
  "Thanks for the quick help at the gate, really appreciate it!",
  "My flight is scheduled as usual. No updates yet."
))

demo_pred <- demo_texts %>%
  bind_cols(predict(final_xgb, demo_texts, type = "prob")) %>%
  bind_cols(predict(final_xgb, demo_texts, type = "class"))

print(demo_pred)
readr::write_csv(demo_pred, "outputs/demo_predictions.csv")

# Save models + session info
saveRDS(final_glmnet, "outputs/final_glmnet_workflow.rds")
saveRDS(final_xgb,    "outputs/final_xgb_workflow.rds")
writeLines(capture.output(sessionInfo()), "outputs/session_info.txt")

cat("\nDONE. Outputs saved in ./outputs\n")

