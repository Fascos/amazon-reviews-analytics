# Part 1: Amazon's Customer Review Analysis

## Overview

This project applies Natural Language Processing (NLP) techniques to Amazon customer reviews to explore customer opinions, identify common language patterns, and prepare the text data for sentiment analysis and text classification.

The dataset contains **997 Amazon reviews** with information about ratings, review text, timestamps, sentiment labels, and review length.

## Objectives

The main objectives of this analysis were to:

* Explore the structure and distribution of the review data.
* Preprocess and clean customer review text.
* Compare stemming and lemmatization.
* Identify the most common words, bigrams, and trigrams.
* Compare language patterns across positive, negative, and neutral reviews.
* Understand what customers discuss about the products.
* Prepare the text data for sentiment analysis and future text classification.

## Dataset

The dataset contains the following key variables:

* `product_id` – Product identifier
* `rating` – Customer rating from 1 to 5
* `review_text` – Full customer review
* `review_summary` – Review summary
* `timestamp` – Review date
* `sentiment_label` – Positive, negative, or neutral sentiment
* `review_length_words` – Number of words in each review

### Key Data Findings

The dataset contains:

* **997 reviews**
* **768 positive reviews**
* **135 negative reviews**
* **94 neutral reviews**

The ratings are also mainly positive, with **5-star reviews being the most common (580 reviews)**.

Review lengths vary considerably, ranging from **5 to 2,079 words**, with most reviews being relatively short.

## Text Preprocessing

The review text was processed using the following steps:

1. Lowercasing
2. URL removal
3. Punctuation removal
4. Tokenization
5. Keeping alphabetic tokens
6. Stopword removal
7. Lemmatization

Stemming and lemmatization were also compared. Stemming sometimes produced shortened or unclear words, while lemmatization produced more meaningful word forms. Therefore, **lemmatization was selected** for the final preprocessing pipeline.

## Frequency Analysis

The most common words included:

* `nook`
* `book`
* `kindle`
* `screen`
* `read`
* `tablet`

These words show that customers frequently discuss **reading devices, books, screens, and product usage**.

Common bigrams included:

* `battery life`
* `touch screen`
* `nook color`
* `nook hd`
* `work great`

Common trigrams included:

* `nook simple touch`
* `micro sd card`
* `barnes noble nook`
* `google play store`

These phrases provide more detailed insight into the products and features discussed by customers.

## Language Patterns by Sentiment

The language patterns differed across sentiment categories.

**Negative reviews** included phrases such as `customer service told`, `credit card`, and `way directional key`, suggesting discussions around problems and difficulties.

**Positive reviews** included phrases such as `work great`, `micro sd card`, and `google play store`, showing discussions around product performance, features, and technology.

**Neutral reviews** included phrases such as `sd card slot`, `web browser`, and `finger across screen`, which were more descriptive of product features and usage.

These patterns help identify **what customers appreciate, what problems they experience, and which product features they discuss**.

## Visualizations

The analysis included visualizations of:

* Rating distribution
* Sentiment distribution
* Review length distribution
* Most common words
* Most common bigrams

These visualizations helped identify the overall structure of the dataset and the main topics appearing in customer reviews.

## Key Insights

The analysis shows that:

* The dataset is largely **positive and high-rated**.
* Customers frequently discuss **Nook and Kindle devices, books, reading, screens, and tablets**.
* Product features such as **battery life, touch screens, SD cards, and web browsers** are commonly discussed.
* Negative reviews contain more language related to **problems and customer difficulties**.
* Positive reviews more often highlight **performance and product features**.
* Neutral reviews tend to provide **descriptive information about product features and usage**.

## Conclusion

The NLP preprocessing and exploratory analysis transformed the raw customer reviews into a cleaner and more consistent form for analysis. Word, bigram, and trigram analysis revealed important product-related topics and differences in language across sentiment categories.

The processed text and existing sentiment labels provide a foundation for the next stage of the NLP task: **building and evaluating models for sentiment analysis or text classification**.

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Matplotlib
* Seaborn
* VS Code
