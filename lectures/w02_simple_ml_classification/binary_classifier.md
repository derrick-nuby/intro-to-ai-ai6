# Binary Classification with MNIST — Detecting the Digit 5

## Objective

In this exercise, you will build a **binary classifier** that can detect whether an image from the MNIST dataset represents the digit **5** or **not-5**.

You’ll work through the process of:

- Loading and preparing the data
- Creating binary target vectors
- Training a simple model using **stochastic gradient descent (SGD)**
- Evaluating your model’s performance

## Background

The MNIST dataset consists of **70,000 grayscale images** of handwritten digits (0–9), each 28×28 pixels, flattened into a vector of 784 features, and each feature simply represents one pixel’s intensity, from 0 (white) to 255 (black). We can see this by inspecting the shape, `X.shape`, `y.shape`. 

This set has been studied so much that it is often called the “hello world” of Machine Learning.

For this task, we’ll turn it into a **binary classification problem**:
- Label **5** as the **positive class** (`True`)
- Label all other digits (i.e 0–4, 6–9) as the **negative class** (`False`)

## Instructions

### 🔹 Step 1: Load the MNIST Dataset

Use `sklearn.datasets.fetch_openml()` to load the MNIST dataset, and extract the `data` and `targets`:

```python
# TODO
```

### 🔹 Step 2: Create the Test Set

Split the data manually:
- Use the **first 60,000 samples** for training
- Use the **last 10,000 samples** for testing

```python
# TODO
```

### 🔹 Step 3: Create Binary Target Vectors

Transform the train and test labels into boolean arrays for a binary "5-detector":

```python
# TODO
```

### 🔹 Step 4: Train a Classifier

Use `SGDClassifier` for its efficiency on large datasets and compatibility with online learning:

```python
# TODO
```

### 🔹 Step 5: Make Predictions

Test the model on new data:

```python
# TODO
```

## Bonus: Evaluate Your Classifier (Optional)

Try adding these steps to explore your model further:

- Use `cross_val_score()` to measure accuracy
- Plot a confusion matrix using `sklearn.metrics`
- Try another classifier like `RandomForestClassifier` for comparison
- Tune hyperparameters (e.g., learning rate, regularization)