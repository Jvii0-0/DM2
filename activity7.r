
# ================================
# SOCIAL MEDIA MINING ACTIVITY
# Kaggle Tweets Dataset
# ================================


# ----------------
# Task 1: Data Loading
# ----------------

# Load required libraries for text mining and visualization
library(tidyverse)
library(tidytext)
library(wordcloud)
library(tm)

# Set working directory where the dataset is stored
setwd("C:/Users/Admin/Desktop/Dm2/Finals/Activity 7")

# Load CSV dataset from Kaggle
data <- read.csv("for_export_dpwh_floodcontrol_translation.csv",
                 stringsAsFactors = FALSE)

# Check column names to understand dataset structure
colnames(data)

# Use the translated column as the main text for analysis
# (This contains the English version from Google Translate)
data$text <- as.character(data$translated)

# Limit dataset to first 50,000 rows for performance and requirement compliance
data <- data[1:min(50000, nrow(data)), ]


# ----------------
# Task 2: Text Cleaning
# ----------------

# Convert text to lowercase for consistency
data$text <- tolower(data$text)

# Remove URLs
data$text <- gsub("http\\S+|www\\S+", "", data$text)

# Remove mentions (@user)
data$text <- gsub("@\\w+", "", data$text)

# Remove hashtags
data$text <- gsub("#\\w+", "", data$text)

# Remove punctuation marks
data$text <- gsub("[[:punct:]]", "", data$text)

# Remove numbers
data$text <- gsub("[[:digit:]]", "", data$text)

# Remove extra white spaces
data$text <- gsub("\\s+", " ", data$text)


# Tokenize text into individual words
tidy_data <- data %>%
  unnest_tokens(word, text)

# Load stopwords (common words like "the", "and", etc.)
data("stop_words")

# Remove stopwords to keep only meaningful words
tidy_data <- tidy_data %>%
  anti_join(stop_words, by = "word")


# ----------------
# Task 3: Word Cloud
# ----------------

# Count frequency of each word
word_count <- tidy_data %>%
  count(word, sort = TRUE)

# Display word cloud (visual representation of frequent words)
wordcloud(words = word_count$word,
          freq = word_count$n,
          max.words = 100,
          colors = rainbow(20))

# Save word cloud as image file for submission
png("wordcloud.png")

wordcloud(words = word_count$word,
          freq = word_count$n,
          max.words = 100,
          colors = rainbow(20))

dev.off()


# ----------------
# Task 4: Sentiment Analysis (Bing Lexicon)
# ----------------

# Load Bing sentiment lexicon (positive/negative words)
bing <- get_sentiments("bing")

# Match dataset words with sentiment lexicon
sentiment_data <- tidy_data %>%
  inner_join(bing, by = "word")


# ----------------
# Task 5: Sentiment Summary
# ----------------

# Count positive and negative words
sentiment_summary <- sentiment_data %>%
  count(sentiment)

# Compute total words and classified words
total_words <- nrow(tidy_data)
classified_words <- nrow(sentiment_data)

# Remaining words are considered neutral
neutral <- total_words - classified_words

# Add neutral category to final summary
sentiment_summary <- rbind(
  sentiment_summary,
  data.frame(sentiment = "neutral", n = neutral)
)

# Print final sentiment distribution
print(sentiment_summary)
