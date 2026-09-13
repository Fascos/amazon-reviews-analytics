# PART 3: Neural Network — MLP Classification

## Overview

This task focused on building a **Multi-Layer Perceptron (MLP)** to classify Amazon customer review ratings. The review text was converted into numerical features using **TF-IDF**, using the 1,000 most important features.

## Model Architecture

The main MLP used two hidden layers:

* **Input layer:** 1,000 TF-IDF features
* **Hidden layer 1:** 128 neurons
* **Hidden layer 2:** 64 neurons
* **Output layer:** 5 neurons for the five rating classes
* **Activation:** Tanh
* **Optimizer:** Adam

Different activation functions were tested. **Tanh performed best**, achieving **61.5% validation accuracy** in the initial activation comparison.

## Hyperparameter Tuning

I experimented with the **learning rate, batch size, number of epochs, dropout, optimizer, and network depth**.

The final best configuration was:

* **128 → 64 neurons**
* **Tanh activation**
* **Adam optimizer**
* **Learning rate:** 0.0001
* **Batch size:** 32
* **Epochs:** 30
* **Dropout:** 0.1

This configuration achieved a **best validation accuracy of 63.0%**, improving on the earlier baseline.

## PyTorch Implementation

I also implemented a basic version of the MLP using **PyTorch**. The PyTorch model achieved **58.0% validation accuracy**, which was lower than the Keras model.

This showed that both frameworks can implement the same basic neural network structure, while Keras provided a simpler training workflow for this task.

## Final Evaluation

The final tuned Keras model achieved:

* **Test Accuracy:** 62%
* **Weighted F1-score:** 0.52
* **Macro F1-score:** 0.27

The model performed best on the majority **rating 5 class**, achieving **99% recall** and a **77% F1-score**.

However, the model struggled with the minority classes. Ratings **2 and 3 had 0% recall**, meaning the model did not correctly identify any of these reviews. Many lower-rated reviews were instead classified as rating 5.

The results show that the model's overall accuracy was strongly influenced by the large number of 5-star reviews.

## Key Findings

The main findings were:

* **Tanh** performed best during the initial activation comparison.
* A **smaller learning rate of 0.0001** improved validation performance.
* Increasing network depth did not improve performance.
* **Adam** performed better than RMSprop.
* A **0.1 dropout rate** slightly improved validation accuracy.
* The model showed signs of **overfitting** during training.
* **Class imbalance** strongly affected performance, especially for minority ratings.
* Accuracy alone did not fully represent performance across all rating classes.

## Conclusion

The MLP was able to classify Amazon review ratings with **moderate performance**, achieving **63.0% best validation accuracy and 62% test accuracy**. However, the model was heavily biased toward the majority 5-star class and struggled to identify lower ratings.

The experiments also showed that increasing model complexity did not necessarily improve performance. Careful hyperparameter tuning provided some improvement, but **class imbalance remained the main limitation**.

## Future Improvements

* Address class imbalance using **class weights or resampling**.
* Use **early stopping** to reduce overfitting.
* Experiment with different TF-IDF features and n-grams.
* Try **L2 regularization** and other dropout settings.
* Test different neural network architectures.
* Explore **LSTM, GRU, or Transformer models** for text classification.
* Evaluate using **macro F1, precision, and recall** alongside accuracy.
