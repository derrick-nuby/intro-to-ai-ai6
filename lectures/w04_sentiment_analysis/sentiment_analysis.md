# Building a Sentiment Analysis Model for the Vaccination Challenge

This tutorial walks you through the process of tackling a sentiment classification problem using **text data from social media**, based on the [*To Vaccinate or Not to Vaccinate challenge*](https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate) on Zindi.

> **Goal**: Predict whether tweets are **pro-vaccination**, **neutral**, or **anti-vaccination** using Natural Language Processing (NLP) and Machine Learning techniques.

## Understanding the Problem

This is a **multi-class sentiment classification** task (3 classes: positive, neutral, negative). You're provided with tweets about vaccinations. Your task is to build a model that predicts the sentiment of unseen tweets.

### Files Overview:
- `train.csv`: Training data (with labels)
- `test.csv`: Data to predict
- `sampleSubmission.csv`: Submission format

### Focus on:
- `safe_text`: The actual tweet content
- `label`: Sentiment category (-1, 0, 1)

## Step 1: Data Exploration

Load the datasets and take a peek at how they looks:

```python
# TODO
```

### Check for:

- Missing values or null values in the **train** and **test** datasets

```python
# TODO
```

- Incorrect class label. Labels must be either **-1, 0, or 1**. Check for rows that have values other than these.

```python
# TODO
```

- Class balance (`label` distribution). You can also use `matplotlib` to view the label distribution.

```python
# TODO
```

- Most frequent and least frequent words (by `label`)

```python
# TODO
```

## Step 2: Preprocessing & Feature Engineering

From the EDA section, you would see that the train and test data has either rows with missing values or where the labels are not -1, 0, or 1. **You should remove this rows**.

```python
# TODO
``` 

Before modeling, clean the text data:
- Remove usernames, links, special characters, punctuations
- Strip extra whitespace
- Convert to lowercase

A minimal code to do the above is below:

```python
import re

def clean_text(text):
    text = str(text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Replace '&' with 'and'
    text = text.replace('&', 'and')

    # Replace punctuation and control characters with spaces
    filters = string.punctuation + '\t\n'
    text = text.translate(str.maketrans({c: ' ' for c in filters}))

    # Normalize spacing and lowercase
    return text.strip().lower()
```
Apply the `clean_text()` function above to the `safe_text` column to create a cleaned text which we will use later on. Store this cleaned text in a new column `cleaned_text`.

```python
# TODO
```
**FROM HERE ONWARD THROUGH THE REST OF THIS GUIDE, YOU SHOULD BE WORKING WITH THE `cleaned_text` COLUMN**.

**Optional**:
- Analyze word count distributions
- Use word clouds to visualize key terms by class


## Step 3: Model Training & Evaluation

Split your data to `X_train`, `y_train` (for training), and `X_valid`, `y_valid` (for validation)

```python
# TODO
```

### Feature Extraction (Vectorization)

You need to convert text into numbers before feeding it into ML models.

**Two common approaches**:
1. **TF-IDF Vectorizer** (great baseline): This gives less importance to words that contain less information and are common in documents, such as ‘the’ and ‘this’ - and to give higher priority to words that have relevant information and appear less frequently.
2. **Transformer embeddings** (e.g., via pre-trained BERT/Roberta)

####  Traditional ML

For this challenge, we are restricting ourselves to traditional ML so we use the **TF-IDF Vectorizer**. You should check how it's used [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

```python
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
```

We have initialized the `TfidfVectorizer` above. Use it to:

1. **Fit and transform** TF-IDF to the training set to create `X_train_tfv`. **Hint**: `.fit_transform()`.
2. Use the fitted vectorizer to **transform** TF-IDF to the validation set to create `X_valid_tfv`. **Hint**: `.transform()`.
3. Use the fitted vectorizer to **transform** TF-IDF to the test set to create `X_test_tfv`. **Hint**: `.transform()`.

```python
# TODO
```

### Training

Use the **fitted and transformed** train datasets from TF-IDF, `X_train_tfv` as inputs to train your model. 

Start with something simple like **Logistic Regression**. Later, try alternative models (e.g., MultinomialNB, Random Forest, LightGBM, XGBoost, etc.)

```python
# TODO
```

### Evaluation (using **RMSE**).

Use the **transformed** validation datasets from TF-IDF, `X_valid_tfv` as inputs to validate your trained/fitted model. 

#### Converting Probabilities to -1, 0, 1 Scale (polarity)

Although, this is a **classification problem**, the competition uses **RMSE**, which is a **regression metric**.

This means:

- You should **output a float value (polarity)**, not just a hard class like -1, 0, or 1
- But that float should still be **centered around the valid labels** (-1, 0, 1)

To do that, we scale the predicted class by its probability:

```python
def process_prediction(preds):
    return [(pred.argmax() - 1) * pred[pred.argmax()] for pred in preds]
```

This allows you to compute a regression-based metric (like RMSE) even from a classifier’s output and gives the most likely class weighted by how confident the model is.

Apply it after prediction for the **validation set**, `X_valid_tfv`:

```python
# TODO
val_predictions_proba = ... # TODO
val_prediction = process_prediction(val_predictions_proba)
```

## Step 4: Generate and Submit Predictions

Repeat the same thing for the **test set**. Use the **transformed** test datasets from TF-IDF, `X_test_tfv` as inputs to test your trained/fitted model.

```python
# TODO
test_predictions_proba = ... # TODO
test_prediction = process_prediction(test_predictions_proba)
```

Prepare predictions in the expected submission format:

```python
sample_sub = pd.read_csv("SampleSubmission.csv")
submission = pd.DataFrame({
    "tweet_id": sample_sub["tweet_id"],
    "label": test_prediction  # Replace with actual model output if different
})
submission.to_csv("submission.csv", index=False)

submission.head()
```
Upload to [Zindi](https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate) and evaluate your performance.

## Bonus: Tips for Improvement

- Tune vectorizers (`ngram`, `min_df`, etc.).
- Try alternative models (e.g., LightGBM, XGBoost)..
- Use ensemble methods.
- Explore weighting or sampling for class imbalance (e.g. [`SMOTE`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)).

## Final Words

This challenge is a great entry point into **text classification** and **NLP pipelines**. Focus on building a clean pipeline, experiment with different model types, and always validate carefully.

> Remember: Your best model isn’t just the one that fits the training data—it generalizes well to unseen examples.

Good luck, and happy modeling! 🚀