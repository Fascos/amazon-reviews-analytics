# Amazon Customer Analytics — NLP, Time Series & Neural Networks

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

## Project structure

Amazon-Reviews-Analytics/
│
├── README.md                 ← overall README
│
├── Task-1-NLP/
│   ├── notebook.ipynb
│   ├── README.md
│   └── amazon_reviews_lab.csv
│
├── Task-2-Time-Series/
│   ├── notebook.ipynb
│   ├── README.md
│   └── amazon_reviews_lab.csv
│
└── Task-3-Neural-Network/
    ├── notebook.ipynb
    ├── README.md
    └── amazon_reviews_lab.csv

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

The processed text provided the foundation for the neural network classification task.

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

# Part 3: Neural Network Classification

## Overview

The third part used the processed review text to build a **Multi-Layer Perceptron (MLP)** for customer rating classification.

The review text was converted into numerical features using **TF-IDF**, using the 1,000 most important features.

## MLP Architecture

The final neural network consisted of:

* **Input:** 1,000 TF-IDF features
* **Hidden layer 1:** 128 neurons
* **Hidden layer 2:** 64 neurons
* **Output:** 5 neurons representing ratings 1–5
* **Activation:** Tanh
* **Optimizer:** Adam
* **Dropout:** 0.1

Different activation functions and network configurations were tested. Tanh achieved the best performance during the initial activation comparison with **61.5% validation accuracy**.

## Hyperparameter Tuning

Different learning rates, batch sizes, epochs, dropout settings, optimizers, and network architectures were tested.

The final best configuration was:

* **128 → 64 neurons**
* **Tanh activation**
* **Adam optimizer**
* **Learning rate: 0.001**
* **Batch size: 32**
* **Epochs: 50**
* **Dropout: 0.1**

This configuration achieved a **best validation accuracy of 61.5%**.

## PyTorch Implementation

A basic version of the MLP was also implemented using PyTorch.

The PyTorch model achieved **58.0% validation accuracy**, which was lower than the Keras model.

This demonstrated that both frameworks could implement the same basic neural network approach, while Keras provided a simpler training workflow for this task.

Here is the corrected **Final Evaluation** section based directly on your updated output:

## Final Evaluation

The final tuned Keras model achieved:

* **Test Accuracy: 52.5%**
* **Weighted F1-score: 0.51**
* **Macro F1-score: 0.29**

The model performed best on the majority **5-star class**, achieving **77% recall** and a **73% F1-score**.

Performance on the lower-rated classes was much weaker. The **2-star class had 0% recall**, meaning the model did not correctly identify any 2-star reviews. The 3-star class also had relatively low performance, with **21% recall**.

The confusion matrix showed that many lower-rated reviews were incorrectly classified as **5-star reviews**, particularly for the 4-star and 3-star classes.

This shows that the model's overall accuracy was strongly influenced by the large number of 5-star reviews in the dataset. The **macro F1-score of 0.29**, compared with the weighted F1-score of 0.51, further highlights the model's difficulty in handling the minority rating classes.

## Key Neural Network Findings

The experiments showed that:

* **Tanh** performed best during the initial activation comparison.
* A learning rate of **0.001** improved validation performance.
* Increasing network depth did not improve performance.
* **Adam** performed better than RMSprop.
* A **0.1 dropout rate** slightly improved validation accuracy.
* The model showed signs of **overfitting** during training.
* **Class imbalance** strongly affected minority-class performance.
* Accuracy alone did not fully represent model performance.

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

* Address class imbalance using **class weights or resampling**.
* Use **early stopping** to reduce overfitting.
* Experiment with different TF-IDF configurations.
* Apply **L2 regularization** and alternative dropout settings.
* Test different neural network architectures.
* Explore **LSTM, GRU, or Transformer models**.
* Evaluate **macro F1, precision, and recall** alongside accuracy.

---

# Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **NLTK**
* **Scikit-learn**
* **Statsmodels**
* **TensorFlow / Keras**
* **PyTorch**
* **VS code**

---

# Overall Conclusion

This project demonstrated how different data science techniques can be applied to the same customer review dataset to answer different business questions.

NLP revealed **what customers were discussing and how language differed across sentiment groups**. Time series analysis showed **how review activity and customer ratings changed over time**, while the neural network explored whether **review text could be used to predict customer ratings**.

The results show that the dataset contains useful information about customer behavior and product experience. However, the analysis also highlighted important challenges, particularly **class imbalance, overfitting, limited data, and uncertainty in forecasting**.

Overall, the project provided practical experience in **text preprocessing, exploratory analysis, time series forecasting, feature engineering, neural network modeling, model evaluation, and interpreting results in a real-world customer analytics context**.

### Author: Fascos Jepleting

### Program: Data Science / Machine Learning

### Institution: Moringa School
