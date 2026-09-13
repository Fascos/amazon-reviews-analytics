# Part 2: Time Series Forecasting: Amazon Reviews and Customer Ratings

## Project Overview

In this project, Amazon review activity and customer ratings were analyzed over time. The analysis examined changes in review volume, trends in monthly average ratings, stationarity, seasonality, and forecasting performance using ARIMA and SARIMA models.

The main forecasting focus was on **monthly average customer ratings**, while review activity was analyzed to provide additional context on how the number of reviews changed over time.

## Objectives

* Analyze review activity over time.
* Examine trends in monthly review volume.
* Analyze changes in monthly average customer ratings.
* Test whether the rating series is stationary.
* Apply transformations such as differencing and log transformation.
* Decompose the rating series into trend, seasonal, and residual components.
* Use ACF and PACF to examine autocorrelation.
* Build and compare ARIMA models.
* Evaluate model performance using residual diagnostics and forecasting metrics.
* Apply SARIMA to account for yearly seasonality.
* Forecast future customer ratings.
* Identify overall trends and possible improvements to the forecasting approach.

## Dataset

The dataset contains **997 Amazon reviews** with information including product ID, rating, review text, review summary, review date, sentiment label, and review length.

The `timestamp` variable was converted to datetime format and used to analyze review activity and create the monthly average rating time series.

## 1. Review Activity Over Time

Monthly review volume was examined to understand how review activity changed throughout the available period.

Review activity was very limited before 2010, with only a few isolated observations. From around 2010 onward, review activity became more consistent and increased over time, reaching higher monthly volumes around 2013–2014.

The increase in review activity provided a more consistent basis for examining customer rating patterns.

## 2. Customer Ratings Over Time

Monthly average customer ratings were calculated to create the main time series used for forecasting.

The ratings generally fluctuated between **3.0 and 5.0**. Earlier periods showed greater variation, while ratings became more stable as review activity increased. By 2013–2014, monthly ratings generally remained around **4.0–4.5**.

A 3-month rolling average was also used to smooth short-term fluctuations and make the underlying rating trend easier to observe.

## 3. Overall Trends in Review Activity and Customer Ratings

Review activity was very low before 2010, with only a few isolated observations. From 2010 onward, review volume increased and became more consistent, reaching higher levels around 2013–2014.

As review activity increased, customer ratings became more stable and generally remained between 4.0 and 4.5. This suggests that higher review volumes were associated with less variation in monthly average ratings, while lower review volumes resulted in greater fluctuations.

**Conclusion:** Overall, review activity increased over time while customer ratings became more stable and consistently positive.

## 4. Stationarity Analysis

The Augmented Dickey-Fuller (ADF) test was used to determine whether the monthly rating series was stationary.

**ADF p-value: 0.5553**

Since the p-value was greater than 0.05, the null hypothesis was not rejected, indicating that the original rating series was not stationary.

First-order differencing was therefore applied to remove the underlying trend and make the series more suitable for ARIMA modeling. A log transformation was also examined to assess its effect on the variance.

## 5. Decomposition and Seasonality

The monthly rating series was decomposed into **trend, seasonal, and residual components**.

The decomposition showed an overall upward trend in ratings and a recurring yearly seasonal pattern. The residual component was mainly centered around zero, indicating that much of the systematic variation was explained by the trend and seasonal components.

The presence of a yearly seasonal pattern supported testing a SARIMA model in addition to standard ARIMA.

## 6. ACF and PACF Analysis

The ACF and PACF plots showed relatively weak autocorrelation, with most spikes falling within the confidence intervals.

Based on these results, relatively low-order ARIMA models were tested and compared using AIC and BIC.

## 7. ARIMA Model Selection

Several ARIMA models were tested to identify a suitable combination of autoregressive, differencing, and moving-average terms.

| Model            |        AIC |        BIC |
| ---------------- | ---------: | ---------: |
| ARIMA(1,1,0)     |     175.02 |     179.43 |
| ARIMA(0,1,1)     |     149.74 |     154.15 |
| ARIMA(1,1,1)     |     150.80 |     157.42 |
| ARIMA(2,1,0)     |     148.88 |     155.50 |
| ARIMA(0,1,2)     |     150.33 |     156.95 |
| ARIMA(3,1,0)     |     148.58 |     157.40 |
| **ARIMA(2,1,1)** | **148.08** | **156.90** |

ARIMA(2,1,1) was selected because it achieved the lowest AIC among the tested models and showed good residual behavior.

## 8. ARIMA Model Diagnostics

The residuals from ARIMA(2,1,1) were mostly centered around zero with relatively stable variance. The Q-Q plot showed that most residuals followed the expected normal pattern, although some deviations appeared at the extreme ends.

The residual ACF showed no significant remaining autocorrelation.

An additional ADF test on the residuals produced a **p-value of 0.00085**, indicating that the residuals were stationary.

Overall, the diagnostic results supported the adequacy of the ARIMA(2,1,1) model.

## 9. ARIMA Forecasting

The data was divided into training and testing sets, with **136 observations used for training and 35 observations used for testing**.

The ARIMA(2,1,1) model achieved:

* **MAE:** 0.3547
* **RMSE:** 0.4230
* **MAPE:** 8.60%

The forecast captured the general level of customer ratings but struggled to reproduce some of the short-term fluctuations and seasonal behavior observed in the actual data.

## 10. SARIMA Model

A SARIMA model was introduced to account for the yearly seasonal pattern identified during decomposition.

The selected structure was:

**SARIMA(2,1,1)(1,1,1,12)**

The first three parameters `(2,1,1)` represent the non-seasonal ARIMA component, while `(1,1,1,12)` represents the seasonal component with a **12-month cycle**.

The seasonal model allows yearly patterns in customer ratings to be considered during forecasting.

## 11. Future Rating Trend

The SARIMA model was fitted using the full historical rating series and used to forecast the next 12 months.

The forecast suggests that monthly customer ratings will remain relatively stable around the current average level, with some recurring seasonal variation.

The 95% confidence interval becomes wider further into the forecast period, showing that uncertainty increases with longer-term predictions.
 
 **The forecast does not indicate a strong upward or downward change in customer ratings. Instead, ratings are expected to remain relatively stable, with some variation caused by seasonal effects and other factors not included in the model.**

## 12. Reflection on Model Parameters and Future Improvements

### Model Parameters

* **Differencing (d):** The original series was non-stationary, so first-order differencing was used to remove the underlying trend.
* **AR and MA terms (p and q):** These terms capture relationships between previous ratings and previous forecasting errors. ARIMA(2,1,1) provided the best overall balance based on AIC and residual diagnostics.
* **Seasonality:** The standard ARIMA forecast became relatively flat because it did not explicitly model the yearly seasonal pattern. SARIMA was therefore introduced with a 12-month seasonal period.

### Future Improvements

* Test additional seasonal ARIMA parameter combinations.
* Compare ARIMA and SARIMA using MAE, RMSE, and MAPE.
* Include review volume as an additional variable.
* Include review length and sentiment information.
* Explore machine learning models for more complex relationships.
* Use more recent and larger datasets.
* Consider external factors such as price changes, promotions, and product updates.

## Conclusion

The analysis showed that Amazon review activity increased over time, particularly from around 2010 onward, while customer ratings became more stable and generally remained positive.

ARIMA(2,1,1) provided a useful baseline forecasting model, while SARIMA was introduced to account for the yearly seasonal pattern identified in the data. The future forecast suggests relatively stable customer ratings, although uncertainty increases further into the forecast period.
