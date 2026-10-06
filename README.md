# Amazon Customer Reviews Analytics: NLP, Time Series Forecasting & Neural Networks

## Project Overview

This project explores Amazon customer reviews using three data science approaches: **Natural Language Processing (NLP), Time Series Forecasting, and Neural Networks**.

The analysis moves from understanding customer review text and opinions, to examining how ratings change over time, and finally to using review text to classify customer ratings.

The project uses **997 Amazon customer reviews** containing product information, ratings, review text, timestamps, sentiment labels, and review length.

---

## Project Objectives

The main objectives were to:

* Explore the structure and distribution of Amazon customer reviews.
* Clean and preprocess customer review text.
* Identify common words and phrases across reviews.
* Compare language patterns across sentiment categories.
* Analyze review activity and customer ratings over time.
* Test stationarity, trends, and seasonality in customer ratings.
* Build and evaluate ARIMA and SARIMA forecasting models.
* Convert review text into numerical TF-IDF features.
* Build and evaluate an MLP neural network for rating classification.
* Evaluate model performance and identify areas for improvement.

---

## Project Structure

```text
Amazon-Reviews-Analytics/
│
├── README.md                 # Overall project documentation
│
├── Task-1-NLP/
│   ├── NLP-Customer-Reviews.ipynb        
│   ├── README.md            
│   └── amazon_reviews_lab.csv
│
├── Task-2-Time-Series/
│   ├── Time-Series-Forecasting.ipynb        
│   ├── README.md             
│   └── amazon_reviews_lab.csv
│
└── Task-3-Neural-Network/
    ├── customer_rating_neural_network        
    ├── README.md            
    └── amazon_reviews_lab.csv
```


# Part 1: Natural Language Processing

## Overview

The first part focused on exploring and preprocessing Amazon customer review text. NLP techniques were used to understand common language patterns, customer sentiment, and the main product features discussed in the reviews.

## Dataset

The dataset contains **997 reviews**, including:

* **768 positive reviews**
* **135 negative reviews**
* **94 neutral reviews**

The ratings were also highly positive, with **5-star reviews being the most common at 580 reviews**.

Review length varied from **5 to 2,079 words**, although most reviews were relatively short.

## Text Preprocessing

The review text was processed using:

1. Lowercasing
2. URL removal
3. Punctuation removal
4. Tokenization
5. Keeping alphabetic tokens
6. Stopword removal
7. Lemmatization

Stemming and lemmatization were compared. Lemmatization produced more meaningful word forms, so it was selected for the final preprocessing pipeline.

## Text Exploration

The most common words included:

* `nook`
* `book`
* `kindle`
* `screen`
* `read`
* `tablet`

Common phrases included:

* `battery life`
* `touch screen`
* `nook color`
* `nook hd`
* `work great`

Some common trigrams included:

* `nook simple touch`
* `micro sd card`
* `barnes noble nook`
* `google play store`

These patterns showed that customers frequently discussed **reading devices, books, screens, battery life, and product features**.

## Sentiment Patterns

The language differed across sentiment categories.

Negative reviews contained phrases related to problems and customer difficulties, such as customer service and credit card issues.

Positive reviews more often discussed successful product performance and useful features.

Neutral reviews tended to describe product features and usage without strongly positive or negative language.

## Key NLP Findings

The NLP analysis showed that:

* The dataset was strongly positive and high-rated.
* Nook and Kindle devices were frequently discussed.
* Battery life, touch screens, SD cards, and web browsers were common topics.
* Negative reviews focused more on problems and difficulties.
* Positive reviews highlighted performance and useful features.
* Neutral reviews were generally more descriptive.

---

# Part 2: Time Series Forecasting

## Overview

The second part analyzed **Amazon review activity and customer ratings over time**.

The main forecasting target was **monthly average customer rating**, while monthly review volume was analyzed to provide additional context.

## Review Activity

Review activity was very limited before 2010. From around 2010 onward, review activity became more consistent and increased over time, reaching higher volumes around 2013–2014.

As review activity increased, monthly customer ratings also became more stable.

## Customer Ratings

Monthly average ratings generally fluctuated between **3.0 and 5.0**.

By 2013–2014, ratings were generally stable around **4.0–4.5**.

A 3-month rolling average was used to smooth short-term fluctuations and make the overall trend easier to observe.

## Stationarity

The Augmented Dickey-Fuller test was used to test whether the original rating series was stationary.

The initial test produced:

**ADF p-value = 0.5553**

Since the p-value was greater than 0.05, the original series was considered non-stationary.

First-order differencing was therefore applied before ARIMA modeling.

## Seasonality

The time series was decomposed into:

* Trend
* Seasonal component
* Residual component

The analysis indicated a recurring **12-month seasonal pattern**, which supported testing a seasonal forecasting model.

## ARIMA Model Selection

Several ARIMA models were compared using AIC and BIC.

The best-performing model was:

**ARIMA(2,1,1)**

It achieved:

* **AIC: 148.08**
* **BIC: 156.90**

The residual ADF test produced a p-value of **0.00085**, indicating that the residuals were stationary.

## ARIMA Forecast Performance

The data was divided into:

* **136 training observations**
* **35 testing observations**

The ARIMA(2,1,1) model achieved:

* **MAE: 0.3547**
* **RMSE: 0.4230**
* **MAPE: 8.60%**

The model captured the general rating level but struggled to reproduce some short-term fluctuations and seasonal behavior.

## SARIMA

Because the analysis identified yearly seasonality, a SARIMA model was introduced:

**SARIMA(2,1,1)(1,1,1,12)**

The 12-month seasonal component allowed the model to account for recurring yearly patterns.

The resulting forecast suggested that customer ratings would remain **relatively stable**, with some seasonal variation. The confidence interval also became wider further into the forecast period, reflecting increasing uncertainty.

## Key Time Series Findings

The analysis showed that:

* Review activity increased substantially over time.
* Customer ratings became more stable as review activity increased.
* The original rating series was non-stationary.
* Differencing was necessary before ARIMA modeling.
* ARIMA(2,1,1) provided a useful baseline model.
* Seasonal patterns supported the use of SARIMA.
* Future ratings were expected to remain relatively stable rather than show a strong upward or downward trend.

---

Perfect. Replace everything from **`# Part 3: Neural Network Classification`** to the end of your README with this:

# Part 3: Neural Network Classification

## Overview

The third part used Amazon review text to build a **Multi-Layer Perceptron (MLP)** for predicting customer ratings from **1 to 5 stars**.

The review text was converted into numerical features using **TF-IDF**, with a maximum of **5,000 features**. The data was split into training and testing sets using stratified sampling to maintain the rating distribution.

## Data Preparation

The dataset was divided into:

* **797 training reviews**
* **200 testing reviews**

The five rating classes were one-hot encoded for the Keras models.

The dataset was highly imbalanced, with **5-star reviews representing 58.17%** of all reviews. This imbalance became an important consideration during model evaluation.

## MLP Architecture

The initial Keras neural network consisted of:

* **Input:** 5,000 TF-IDF features
* **Hidden layer 1:** 64 neurons
* **Hidden layer 2:** 32 neurons
* **Output:** 5 neurons representing ratings 1–5
* **Activation:** ReLU
* **Dropout:** 0.3
* **Optimizer:** Adam
* **Learning rate:** 0.001
* **Epochs:** 20

The initial ReLU model achieved **54.5% test accuracy**. Training accuracy reached 100%, while validation accuracy remained lower, showing clear signs of overfitting.

## Activation Function Experiments

Three activation functions were compared:

* ReLU
* Tanh
* Sigmoid

ReLU and Tanh achieved **54.5% test accuracy**, while Sigmoid achieved **58.0%**.

However, ReLU achieved the strongest validation performance during the activation comparison, reaching **63.75% validation accuracy**. Therefore, **ReLU** was selected for further experiments.

## Keras and PyTorch Comparison

The same basic MLP structure was also implemented using **PyTorch**.

The comparison showed that:

* Keras uses a simpler Sequential model structure.
* Keras provides a built-in training process using `model.fit()`.
* PyTorch requires a custom model class and manual training loop.
* Keras achieved the higher validation accuracy in the framework comparison.

The Keras implementation was therefore selected for the final model.

## Model Optimization

Several experiments were performed to improve model performance and reduce overfitting.

The experiments included:

* Class weighting
* Learning rate
* Batch size
* Number of hidden layers
* Number of neurons
* Dropout rate
* Optimizer
* Number of epochs

Class weighting improved test accuracy from **54.5% to 57.5%**, showing that addressing class imbalance could provide a modest improvement.

The hyperparameter experiments selected:

* **Learning rate:** 0.01
* **Batch size:** 16
* **Architecture:** 64 neurons in one hidden layer
* **Dropout:** 0.5
* **Optimizer:** Adam
* **Epochs:** 20

The final model was therefore simplified from the original two-hidden-layer architecture.

## Final Model

The final tuned Keras model used:

**5000 → 64 → 5**

with:

* **TF-IDF features:** 5,000
* **Hidden layer:** 64 neurons
* **Hidden activation:** ReLU
* **Output activation:** Softmax
* **Optimizer:** Adam
* **Learning rate:** 0.01
* **Dropout:** 0.5
* **Batch size:** 16
* **Epochs:** 20

## Final Evaluation

The final tuned Keras model achieved:

* **Test Accuracy: 55.5%**
* **Test Loss: 2.1075**
* **Best Validation Accuracy: 61.25%**
* **Macro F1-score: 0.31**
* **Weighted F1-score: 0.51**

The model performed best on the majority **5-star class**, achieving:

* **Precision:** 0.65
* **Recall:** 0.83
* **F1-score:** 0.73

Performance on the lower rating classes was considerably weaker. The **3-star class had an F1-score of only 0.08**, showing that the model struggled to distinguish some of the less frequent ratings.

The confusion matrix also showed that many lower-rated reviews were incorrectly classified as higher ratings, particularly as **5-star reviews**.

These results show that the model was strongly influenced by the class imbalance in the dataset. The difference between the macro F1-score of **0.31** and weighted F1-score of **0.51** further demonstrates that performance was much stronger for the majority class than for the minority classes.

## Key Neural Network Findings

The experiments showed that:

* ReLU provided the strongest validation performance during the activation comparison.
* Class weighting provided a modest improvement in test accuracy.
* Simpler network architectures performed better than deeper networks.
* A dropout rate of 0.5 provided the strongest validation performance during dropout tuning.
* Adam performed better than RMSprop.
* 20 epochs provided the best result during epoch tuning.
* The final model still showed signs of overfitting.
* Class imbalance strongly affected predictions for lower rating classes.
* Accuracy alone was not enough to describe model performance.

---

# Overall Key Findings

Across the three parts of the project, several important patterns emerged.

First, the NLP analysis showed that Amazon reviews were **strongly positive**, with customers frequently discussing devices, books, screens, battery life, and other product features.

Second, the time series analysis showed that **review activity increased over time while customer ratings became more stable**, generally remaining around 4.0–4.5 in the later period.

Finally, the neural network demonstrated that review text contains useful information for predicting customer ratings, but **class imbalance made it difficult to accurately identify lower ratings**.

The three tasks therefore provide different perspectives on the same dataset:

**NLP → Understand what customers are saying**

**Time Series → Understand how customer ratings change over time**

**Neural Network → Predict customer ratings from review text**

---

# Future Improvements

Several improvements could make the analysis more robust.

### NLP

* Use larger datasets with more balanced sentiment classes.
* Explore additional text features and n-grams.
* Apply more advanced language models for deeper text understanding.

### Time Series

* Test additional SARIMA configurations.
* Compare ARIMA and SARIMA using the same forecasting metrics.
* Include review volume, sentiment, and review length as additional variables.
* Explore machine learning models for time series forecasting.
* Include external factors such as product updates, promotions, and price changes.

### Neural Networks

* **Use early stopping** to stop training when validation performance stops improving.
* **Apply L2 regularization** to reduce model complexity and overfitting.
* **Use oversampling** to increase representation of minority rating classes.
* **Improve TF-IDF features** by testing different feature sizes and n-gram settings.
* **Use word embeddings** such as Word2Vec or GloVe.
* **Explore LSTM or GRU models** to capture word sequence and context.
* **Explore transformer models** such as BERT for deeper contextual understanding.

---

# Overall Conclusion

This project demonstrated how different data science techniques can be applied to the same customer review dataset to answer different business questions.

NLP revealed **what customers were discussing and how language differed across sentiment groups**. Time series analysis showed **how review activity and customer ratings changed over time**, while the neural network explored whether **review text could be used to predict customer ratings**.

The results show that the dataset contains useful information about customer behavior and product experience. However, the analysis also highlighted important challenges, particularly **class imbalance, overfitting, limited data, and uncertainty in forecasting**.

Overall, the project provided practical experience in **text preprocessing, exploratory analysis, time series forecasting, TF-IDF feature engineering, neural network modeling, framework comparison, hyperparameter tuning, model evaluation, and interpreting results in a real-world customer analytics context**.

### Author: Fascos Jepleting

### Program: Data Science / Machine Learning

### Institution: Moringa School

