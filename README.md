# Homeworks
---
## Homework 1
### Getting the Data
Use the **Car Fuel Efficiency** dataset.  
**Download**:  
- 📥 Get it from [here](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv).  
- Via terminal:  
  ```bash
  wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
  ```  
- Or download via browser ("Save as...").  
- Load the dataset with Pandas.

---

## Q2. Records Count
**Question**: How many records are in the dataset?  
**Options**: 4704, 8704, 9704, 17704  
**Instructions**: Use `df.shape[0]` to count rows.

---

## Q3. Fuel Types
**Question**: How many unique fuel types are in the dataset?  
**Options**: 
1, 
2,
3, 
4  
**Instructions**: Use `df['fuel_type'].nunique()` to find unique values.

---

## Q4. Missing Values
**Question**: How many columns have missing values?  
**Options**:
0, 
1, 
2, 
3, 
4
**Instructions**: Check with `df.isna().sum()`.

---

## Q5. Max Fuel Efficiency
**Question**: What's the maximum fuel efficiency of cars from **Asia**?  
**Options**: 
13.75, 
23.75, 
33.75, 
43.75  
**Instructions**: Filter for Asia and find the max fuel efficiency.

---

## Q6. Median Value of Horsepower
**Question**: Does the median of `horsepower` change after filling missing values?  
**Steps**:  
1. Calculate the median of `horsepower`.  
2. Find the most frequent value (mode) of `horsepower`.  
3. Use `fillna` to replace missing values with the mode.  
4. Recalculate the median.  
**Options**:
Yes, it increased,
Yes, it decreased,
No  
**Instructions**: Use `median()`, `mode()`, and `fillna()`.

---

## Q7. Sum of Weights
**Question**: What's the sum of elements in the weight vector `w`?  
**Steps**:  
1. Select cars from Asia.  
2. Select `vehicle_weight` and `model_year` columns.  
3. Take the first 7 rows.  
4. Convert to NumPy array (`X`).  
5. Compute `XTX = X.T @ X`.  
6. Invert `XTX` to get `XTX_inv`.  
7. Create array `y = [1100, 1300, 800, 900, 1000, 1100, 1200]`.  
8. Compute `w = (XTX_inv @ X.T) @ y`.  
9. Sum all elements in `w`.  
**Note**: This implements linear regression.  
**Options**:
0.051,
0.51,
5.1,
51  
**Instructions**: Use NumPy for matrix operations (`np.array`, `np.linalg.inv`, `@`).

---

## Notes
- Ensure all dependencies are installed.  
- Verify calculations to match options.  
- For Q7, ensure proper NumPy array formatting for matrix operations.
---
## Homework 2

## Homework: Car Fuel Efficiency Prediction

> **Note**: Sometimes your answer may not match one of the options exactly. That's fine. Select the option that's closest to your solution.

## Dataset

For this homework, we'll use the Car Fuel Efficiency dataset. Download it from [here](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv).

You can do it with `wget`:
```bash
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
```

The goal is to create a regression model for predicting car fuel efficiency (column `'fuel_efficiency_mpg'`).

## Preparing the Dataset

Use only the following columns:
- `'engine_displacement'`
- `'horsepower'`
- `'vehicle_weight'`
- `'model_year'`
- `'fuel_efficiency_mpg'`

## EDA

- Check the `fuel_efficiency_mpg` variable. Does it have a long tail?

## Question 1

There's one column with missing values. What is it?

- `'engine_displacement'`
- `'horsepower'`
- `'vehicle_weight'`
- `'model_year'`

## Question 2

What's the median (50% percentile) for the variable `'horsepower'`?

- 49
- 99
- 149
- 199

## Prepare and Split the Dataset

- Shuffle the dataset (the filtered one created above) using seed `42`.
- Split the data into train/validation/test sets with a 60%/20%/20% distribution.
- Use the same code as in the lectures.

## Question 3

- We need to handle missing values for the column identified in Q1.
- Options: fill with 0 or with the mean of the variable.
- For each option, train a linear regression model without regularization using the code from the lessons.
- Compute the mean using only the training dataset.
- Evaluate the models on the validation dataset and compare the RMSE of each option.
- Round RMSE scores to 2 decimal digits using `round(score, 2)`.
- Which option gives better RMSE?

Options:
- With 0
- With mean
- Both are equally good

## Question 4

- Train a regularized linear regression model.
- Fill missing values with 0.
- Try different values of `r` from: `[0, 0.01, 0.1, 1, 5, 10, 100]`.
- Evaluate the model on the validation dataset using RMSE.
- Round RMSE scores to 2 decimal digits.
- Which `r` gives the best RMSE? If multiple options yield the same RMSE, select the smallest `r`.

Options:
- 0
- 0.01
- 1
- 10
- 100

## Question 5

- We used seed 42 for splitting. Let's explore how the seed affects the score.
- Try seed values: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.
- For each seed, split the data into train/validation/test with 60%/20%/20% distribution.
- Fill missing values with 0 and train a model without regularization.
- Evaluate the model on the validation dataset and collect RMSE scores.
- Compute the standard deviation of the RMSE scores using `np.std`.
- Round the result to 3 decimal digits (`round(std, 3)`).

What's the value of std?

- 0.001
- 0.006
- 0.060
- 0.600

> **Note**: Standard deviation indicates how varied the values are. A low standard deviation means the values are similar, indicating a *stable* model. A high standard deviation means the values differ.

## Question 6

- Split the dataset as before, using seed 9.
- Combine the train and validation datasets.
- Fill missing values with 0 and train a model with `r=0.001`.
- What's the RMSE on the test dataset?

Options:
- 0.15
- 0.515
- 5.15
- 51.5


# Homework 04: Bank Marketing Dataset Analysis

**Note**: Sometimes your answer may not match one of the options exactly. That's fine. Select the option that's closest to your solution.

In this homework, we will use the **Bank Marketing dataset** for lead scoring. [Download it here](<https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv>).

The target variable for the classification task is `converted` - whether the client signed up to the platform or not.

## Data Preparation
- **Check for missing values** in the features.
- If missing values exist:
  - For **categorical features**, replace them with `'NA'`.
  - For **numerical features**, replace them with `0.0`.
- Split the data into three parts: **train/validation/test** with a **60%/20%/20%** distribution using the `train_test_split` function with `random_state=1`.

## Question 1: ROC AUC Feature Importance
ROC AUC can be used to evaluate the feature importance of numerical variables. For each numerical variable:
- Use it as the score (aka prediction) and compute the AUC with the `converted` variable as the ground truth.
- Use the **training dataset** for this.
- If the AUC is < 0.5, invert the variable by adding a negative sign (e.g., `-df_train['balance']`).
  - AUC < 0.5 indicates the variable is negatively correlated with the target. Negating the variable converts negative correlation to positive.

**Which numerical variable (among the following 4) has the highest AUC?**
- `lead_score`
- `number_of_courses_viewed`
- `interaction_count`
- `annual_income`

## Question 2: Training the Model
- Apply **one-hot encoding** using `DictVectorizer`.
- Train a **logistic regression model** with the following parameters:
  ```python
  LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
  ```
- **What's the AUC of this model on the validation dataset?** (Round to 3 digits)
  - 0.32
  - 0.52
  - 0.72
  - 0.92

## Question 3: Precision and Recall
- Evaluate the model on all thresholds from **0.0 to 1.0** with a step of **0.01**.
- For each threshold, compute **precision** and **recall**.
- Plot the precision and recall curves.
- **At which threshold do the precision and recall curves intersect?**
  - 0.145
  - 0.345
  - 0.545
  - 0.745

## Question 4: F1 Score
Precision and recall are conflicting metrics—when one increases, the other often decreases. The **F1 score** combines both using the formula:

$$F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$

Where \( P \) is precision and \( R \) is recall.

- Compute the F1 score for all thresholds from **0.0 to 1.0** with an increment of **0.01**.
- **At which threshold is the F1 score maximal?**
  - 0.14
  - 0.34
  - 0.54
  - 0.74

## Question 5: 5-Fold Cross-Validation
- Use the `KFold` class from Scikit-Learn to evaluate the model on **5 different folds**:
  ```python
  KFold(n_splits=5, shuffle=True, random_state=1)
  ```
- For each fold:
  - Split the data into train and validation.
  - Train the model with:
    ```python
    LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    ```
  - Use **AUC** to evaluate the model on the validation set.
- **How large is the standard deviation of the scores across different folds?**
  - 0.0001
  - 0.006
  - 0.06
  - 0.36

## Question 6: Hyperparameter Tuning
- Use **5-fold cross-validation** to find the best parameter `C`.
- Iterate over the following `C` values: `[0.000001, 0.001, 1]`.
- Initialize `KFold` with the same parameters as in Question 5.
- Use these parameters for the model:
  ```python
  LogisticRegression(solver='liblinear', C=C, max_iter=1000)
  ```
- Compute the **mean score** and **standard deviation** (round both to 3 decimal digits).
- **Which `C` leads to the best mean score?**
  - 0.000001
  - 0.001
  - 1

**Note**: If there are ties, select the score with the lowest standard deviation. If ties persist, select the smallest `C`.

