# Part 3: Neural Network Modeling for Customer Rating Prediction

## Predicting Customer Ratings from Review Text Using TF-IDF and Neural Networks

## Overview

This project focuses on predicting customer ratings from Amazon review text using neural networks.

The project uses customer review text as the input and ratings from **1 to 5 stars** as the target. The review text was converted into numerical features using **TF-IDF**, which were then used to train neural network models.

The main focus of this lab was to explore neural network modeling, compare **TensorFlow/Keras and PyTorch**, experiment with different model settings, and evaluate how well the final model could predict customer ratings.

## Dataset

The dataset contains **997 Amazon customer reviews** with the following main variables:

* `review_text` – Customer review text used as the input.
* `rating` – Customer rating from 1 to 5 used as the target.
* `review_summary` – Short summary of the review.
* `review_length_words` – Number of words in the review.
* Additional fields containing product and review information.

The dataset had:

* **997 reviews**
* **5 rating classes**
* **No missing values**
* **No duplicate rows**

The ratings were imbalanced, with **5-star reviews making up 58.17%** of the dataset. This imbalance became an important consideration during model evaluation.

## Data Preparation

The review text was separated from the rating target.

The dataset was split into:

* **80% training data:** 797 reviews
* **20% testing data:** 200 reviews

Stratified splitting was used to maintain a similar rating distribution between the training and testing sets.

### TF-IDF

TF-IDF was used to convert the review text into numerical features.

The vectorizer was limited to **5,000 features**.

The TF-IDF vectorizer was fitted only on the training reviews and then applied to the test reviews.

The rating values were also one-hot encoded for the Keras classification model.

## Neural Network Modeling

A basic Multi-Layer Perceptron (MLP) was developed for the five-class rating prediction problem.

The initial architecture was:

**5000 → 64 → 32 → 5**

The hidden layers used ReLU activation, while the output layer used Softmax.

The model was trained using:

* Adam optimizer
* Learning rate of 0.001
* Batch size of 32
* 20 epochs
* Dropout of 0.3

The initial ReLU model achieved **54.5% test accuracy** but showed strong overfitting, with training accuracy reaching 100%.

## Activation Function Experiments

Three activation functions were tested:

* ReLU
* Tanh
* Sigmoid

ReLU and Tanh both achieved **54.5% test accuracy**, while Sigmoid achieved **58.0%**.

However, ReLU achieved the strongest validation performance during the activation experiments, reaching **63.75% validation accuracy**.

Therefore, **ReLU + Softmax** was selected for the remaining experiments.

## Keras and PyTorch Comparison

The same basic MLP architecture was also implemented using PyTorch.

The comparison showed that:

* Keras uses a simpler Sequential model structure.
* Keras provides a built-in training workflow using `model.fit()`.
* PyTorch requires a more explicit model class and training loop.
* Keras achieved the higher validation accuracy in the comparison.

Based on the results and simpler workflow, **Keras was selected for the final model**.

## Model Optimization

Several experiments were performed to improve the model and reduce overfitting.

### Class Weighting

Class weighting was tested to address the imbalance between rating classes.

The weighted model achieved:

* **57.5% test accuracy**
* **63.13% validation accuracy**

This improved test accuracy compared with the original ReLU model's 54.5%.

### Hyperparameter Experiments

The following settings were tested:

* Learning rate
* Batch size
* Number of hidden layers
* Number of neurons
* Dropout rate
* Optimizer
* Number of epochs

The experiments showed that:

* Learning rate **0.01** performed best in the learning-rate experiment.
* Batch size **16** achieved the highest validation accuracy in the batch-size experiment.
* A simpler network performed better than larger networks.
* **0.5 dropout** achieved the highest validation accuracy in the dropout experiment.
* Adam performed better than RMSprop.
* **20 epochs** provided the best overall result in the epoch experiment.

## Final Model

The final selected model used the following architecture:

**5000 → 64 → 5**

### Final settings

* **Input features:** 5,000 TF-IDF features
* **Hidden neurons:** 64
* **Hidden activation:** ReLU
* **Output activation:** Softmax
* **Optimizer:** Adam
* **Learning rate:** 0.01
* **Dropout:** 0.5
* **Batch size:** 16
* **Epochs:** 20

## Final Results

The final tuned Keras model achieved:

* **Best validation accuracy:** 61.25%
* **Test accuracy:** 55.5%
* **Test loss:** 2.1075

The classification report showed that the model performed best on the majority **5-star rating**, with:

* Precision: **0.65**
* Recall: **0.83**
* F1-score: **0.73**

Performance on the lower rating classes was considerably weaker. Rating 3 had an F1-score of only **0.08**.

The overall:

* **Macro F1-score:** 0.31
* **Weighted F1-score:** 0.51

The confusion matrix also showed that many lower ratings were incorrectly predicted as higher ratings, particularly 5 stars.

## Key Findings

The experiments showed that a neural network can learn patterns from customer review text, but the current model has difficulty generalizing to unseen reviews.

The main limitation was the strong imbalance in the rating classes. Because 5-star reviews represented the majority of the dataset, the model became much better at predicting 5-star reviews than the less frequent ratings.

The final model also showed clear signs of overfitting, with training accuracy reaching almost 100% while validation accuracy remained much lower.

## Future Improvements

The following improvements were not implemented in this lab but could be explored in future work:

1. **Use early stopping** to stop training when validation performance stops improving.
2. **Apply L2 regularization** to reduce model complexity and overfitting.
3. **Use oversampling** to increase representation of minority rating classes.
4. **Improve TF-IDF features** by testing different feature sizes and n-gram settings.
5. **Use word embeddings** such as Word2Vec or GloVe.
6. **Explore LSTM or GRU models** to capture word sequence and context.
7. **Explore transformer models** such as BERT for deeper contextual understanding.

## Conclusion

This lab demonstrated the complete process of building a neural network for customer rating prediction, from text preprocessing and TF-IDF feature extraction to model development, framework comparison, optimization, and final evaluation.

The final model achieved **55.5% test accuracy**, but its performance was strongly affected by class imbalance and overfitting. The experiments also demonstrated the importance of using validation performance and class-specific metrics rather than relying on accuracy alone when evaluating an imbalanced classification problem.
