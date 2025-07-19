# Fraud Detection in Electricity and Gas Consumption

Welcome to today's Machine Learning challenge!

In this exercise, you'll take on the role of a data scientist working for a utility company to **detect fraudulent behavior** using real-world client and invoice data.

## Objective

Your goal is to **build a classification model** that can predict the probability that a customer is involved in **fraudulent activity**, based on historical invoice and client data.

You will be submitting your results to the [Zindi Challenge Platform](https://zindi.africa/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge) and ranking on the leaderboard 🥳.

## Dataset Downloads

Before starting, download the required datasets and unzip them into your working directory.

| File                 | Download Link |
|----------------------|----------------|
| `train.zip`   | [Download](https://drive.google.com/uc?export=download&id=1UK90pfuTM8i0RYDbSScX_ivUTH7TNXKp) |
| `test.zip`  | [Download](https://drive.google.com/uc?export=download&id=1Wx0bbuUIQy1AW3eH3diE6s8zwkl2wfNq) |
| `SampleSubmission.csv` | [Download](https://drive.google.com/uc?export=download&id=1z7SFAbe4bXrIRomxL9yU4OupEL2Nxeyr) |

## Data Overview

You are provided with the following files:

- `Client_train.csv`: Information about clients in the training set.
- `Invoice_train.csv`: Invoices for those clients.
- `Client_test.csv`: Client info for prediction (no labels).
- `Invoice_test.csv`: Invoice data for the test clients.
- `SampleSubmission.csv`: Format for your final submission.

Each client is identified by a unique `client_id`. Your task is to **combine relevant data, engineer meaningful features, and train a model** to predict fraud.

## Suggested Workflow (Creative freedom encouraged!)

### 1. Data exploration and understanding
- Visualize the data (statistically - using `pandas`). What kinds of variables/features are available in each file?

```python
# TODO
```
- What’s the shape and structure of each dataset?

```python
# TODO
```
- Its important to understand each column and which data it represents. Next, use the pandas `pd.describe()` or `pd.info()` function to view statistical details of the numeric variables.

```python
# TODO
```
- Are there any missing values? Check for missing values.

```python
# TODO
```
- Find out how many people are involved in fraudulent activities.

  Here you will focus on the `target` column that classifies between fraud(1) and not Fraud(0).

```python
# TODO
```
**Bonus**: You can also use  `matplotlib` to visualise the `target` distribution

```python
#Visualize fraudulent activities
fraudactivities = client_train.groupby(['target'])['client_id'].count()
plt.bar(x=fraudactivities.index, height=fraudactivities.values, tick_label = [0,1])
plt.title('Fraud - Target Distribution')
plt.show()
```

### 2. Feature Engineering

**IMPORTANT**: In this section, **You must mirror all transformations done on the training set to the test set** — including encoding, feature creation, and aggregations, etc — to ensure consistency.

How can you combine or transform features to reveal fraud tendencies? Consider:

- From our data explanatory section, we know that `invoice_date` is an object type which we can change to be represented as a `date_time` object using `pd.to_datetime`:

```python
# TODO
```
- Handle Inconsistent Data Types

  The `counter_statue` column appears with **different data types** in the train and test sets. `counter_statue` is an `object` in train but an `int64` in test.

  To ensure compatibility: **Force consistent data types** by treating `counter_statue` as a categorical feature by converting both to `string`

```python
# TODO
```

- Encoding categorical features

  **Label encoding** is the process of converting the labels into numerical form.

  The `Counter_type` column in the **invoice data** has 2 unique values the **ELEC** and **GAZ**, they should be mapped to `0` or `1`.

  Hint: you can use `scikit-learn`'s `LabelEncoder` or do it manually by `.map()`.

```python
# TODO
```
  Currently, the `client_catg` and `district` columns in the **client data** are of `int` data type, we  need to convert them to a categorical object (`str`) to use them on our machine learning algorithm.

```python
# TODO
```
- Aggregating **invoice data** per client

  Since the original **invoice data** has multiple rows per client (one per invoice), we want one row per client, summarizing their invoice behavior.

Aggregate the invoice data at the `client_id` level.

```python
  def aggregate_by_client_id(df):
    num_cols = df.select_dtypes(include='number').columns.tolist()
    num_cols = [col for col in num_cols if col != 'client_id']

    aggs = {col: ['mean'] for col in num_cols}
    
    agg_df = df.groupby('client_id').agg(aggs)
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)

    # Add transaction count
    count_df = df.groupby('client_id').size().reset_index(name='transactions_count')

    return pd.merge(count_df, agg_df, on='client_id', how='left')
```
Use the `aggregate_by_client_id()` function to create your aggregated invoice data

```python
# TODO
```
- Next, we need to merge our aggregated invoice data with our client data to form our `train` data:

```python
# TODO
```
- Drop unnecessary columns in our (train and test) data set

  The variables `client_id` and `creation_date` don't affect customers' probability of being involved in fraudulent activities.

```python
# TODO
```

After the feature engineering section, the shape of both the train data and the test data should be:

```python
train.shape, test.shape
```
```bash
((135493, 18), (58069, 17))
```

### 3. Modeling
- Choose a classification model to start with (e.g., Logistic Regression, Random Forest, LightGBM, XGBoost, etc.)

  Remember to first extract your `x_train` and `y_train` from your `train` data

```python
# TODO
```
**Bonus**:
- Evaluate your model using the **AUC** metric
- Don’t forget to tune your hyperparameters!

```python
# TODO
```
### 4. Make Predictions
- Predict fraud probabilities for each client in the test set

```python
# TODO
```

- Format your predictions according to `SampleSubmission.csv`

```python
sample_sub = pd.read_csv("SampleSubmission.csv")
sub_client_id = sample_sub['client_id']
submission = pd.DataFrame(
    {
        'client_id': sub_client_id,
        'target': preds # assuming your predictions are saved in the `preds` variable. Feel free to change this if otherwise
    }
)

submission.head()
```

### 5. Submit
- Export your submission as `submission.csv`

```python
# TODO
```
- Upload it to the [Zindi Challenge Page](https://zindi.africa/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge)
- Check your score and iterate if time allows!


## Final Submission Requirements

- A `submission.csv` file with two columns: `client_id` and `target` (fraud probability).
- Optional: A one-slide summary of your approach and feature choices.


## Tips

- Be thoughtful about how you combine the datasets.
- Pay attention to the time-based elements in the data.
- Keep your model simple to start—improve incrementally.
- There are **no missing values**—but are there inconsistent or redundant features?
- Visualizations may help spark ideas!


Good luck, and have fun hunting fraud! 🚀
