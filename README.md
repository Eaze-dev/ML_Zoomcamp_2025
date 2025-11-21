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


## Homework 06

> **Note**: sometimes your answer doesn't match one of  
> the options exactly. That's fine.  
> Select the option that's closest to your solution.  
> If it's exactly in between two options, select the higher value.

### Dataset

In this homework, we continue using the fuel efficiency dataset.  
Download it from [here](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv).

You can do it with `wget`:

```bash
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
```

The goal of this homework is to create a regression model for predicting the car fuel efficiency (column `'fuel_efficiency_mpg'`).

### Preparing the dataset 

Preparation:

* Fill missing values with zeros.
* Do train/validation/test split with 60%/20%/20% distribution. 
* Use the `train_test_split` function and set the `random_state` parameter to 1.
* Use `DictVectorizer(sparse=True)` to turn the dataframes into matrices.

---

## Question 1

Let's train a decision tree regressor to predict the `fuel_efficiency_mpg` variable. 

* Train a model with `max_depth=1`.

Which feature is used for splitting the data?

- `'vehicle_weight'`
- `'model_year'`
- `'origin'`
- `'fuel_type'`

---

## Question 2

Train a random forest regressor with these parameters:

* `n_estimators=10`
* `random_state=1`
* `n_jobs=-1` (optional - to make training faster)

What's the RMSE of this model on the validation data?

- 0.045
- 0.45
- 4.5
- 45.0

---

## Question 3

Now let's experiment with the `n_estimators` parameter

* Try different values of this parameter from 10 to 200 with step 10.
* Set `random_state` to `1`.
* Evaluate the model on the validation dataset.

After which value of `n_estimators` does RMSE stop improving?  
Consider 3 decimal places for calculating the answer.

- 10
- 25
- 80
- 200

> If it doesn't stop improving, use the latest iteration number in your answer.

---

## Question 4

Let's select the best `max_depth`:

* Try different values of `max_depth`: `[10, 15, 20, 25]`
* For each of these values,
  * try different values of `n_estimators` from 10 till 200 (with step 10)
  * calculate the mean RMSE 
* Fix the random seed: `random_state=1`

What's the best `max_depth`, using the mean RMSE?

- 10
- 15
- 20
- 25

---

## Question 5

We can extract feature importance information from tree-based models. 

At each step of the decision tree learning algorithm, it finds the best split.  
When doing it, we can calculate "gain" - the reduction in impurity before and after the split.  
This gain is quite useful in understanding what are the important features for tree-based models.

In Scikit-Learn, tree-based models contain this information in the  
[`feature_importances_`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.feature_importances_) field. 

For this homework question, we'll find the most important feature:

* Train the model with these parameters:
  * `n_estimators=10`,
  * `max_depth=20`,
  * `random_state=1`,
  * `n_jobs=-1` (optional)
* Get the feature importance information from this model

What's the most important feature (among these 4)? 

- `vehicle_weight`
- `horsepower`
- `acceleration`
- `engine_displacement`

---

## Question 6

Now let's train an XGBoost model! For this question, we'll tune the `eta` parameter:

* Install XGBoost
* Create DMatrix for train and validation
* Create a watchlist
* Train a model with these parameters for 100 rounds:

```python
xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}
```

Now change `eta` from `0.3` to `0.1`.

Which eta leads to the best RMSE score on the validation dataset?

- 0.3
- 0.1
- Both give equal value
- 
``

## Homework 08

> **Note**: it's very likely that in this homework your answers won't match 
> the options exactly. That's okay and expected. Select the option that's
> closest to your solution.
> If it's exactly in between two options, select the higher value.

### Dataset

In this homework, we'll build a model for classifying various hair types. 
For this, we will use the Hair Type dataset that was obtained from 
[Kaggle](https://www.kaggle.com/datasets/kavyasreeb/hair-type-dataset) 
and slightly rebuilt.

You can download the target dataset for this homework from 
[here](https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip):

```bash
wget https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip
unzip data.zip
```

In the lectures we saw how to use a pre-trained neural network. In the homework, we'll train a much smaller model from scratch. 

We will use PyTorch for that.

You can use Google Colab or your own computer for that.

### Data Preparation

The dataset contains around 1000 images of hairs in the separate folders 
for training and test sets. 

### Reproducibility

Reproducibility in deep learning is a multifaceted challenge that requires attention 
to both software and hardware details. In some cases, we can't guarantee exactly the same results during the same experiment runs.

Therefore, in this homework we suggest to set the random number seed generators by:

```python
import numpy as np
import torch

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Also, use PyTorch of version 2.8.0 (that's the one in Colab).

### Model

For this homework we will use Convolutional Neural Network (CNN). We'll use PyTorch.

You need to develop the model with following structure:

* The shape for input should be `(3, 200, 200)` (channels first format in PyTorch)
* Next, create a convolutional layer (`nn.Conv2d`):
    * Use 32 filters (output channels)
    * Kernel size should be `(3, 3)` (that's the size of the filter)
    * Use `'relu'` as activation 
* Reduce the size of the feature map with max pooling (`nn.MaxPool2d`)
    * Set the pooling size to `(2, 2)`
* Turn the multi-dimensional result into vectors using `flatten` or `view`
* Next, add a `nn.Linear` layer with 64 neurons and `'relu'` activation
* Finally, create the `nn.Linear` layer with 1 neuron - this will be the output
    * The output layer should have an activation - use the appropriate activation for the binary classification case

As optimizer use `torch.optim.SGD` with the following parameters:

* `torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.8)`


### Question 1

Which loss function you will use?

* `nn.MSELoss()`
* `nn.BCEWithLogitsLoss()`
* `nn.CrossEntropyLoss()`
* `nn.CosineEmbeddingLoss()`

(Multiple answered can be correct, so pick any)


### Question 2

What's the total number of parameters of the model? You can use `torchsummary` or count manually. 

In PyTorch, you can find the total number of parameters using:

```python
# Option 1: Using torchsummary (install with: pip install torchsummary)
from torchsummary import summary
summary(model, input_size=(3, 200, 200))

# Option 2: Manual counting
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
```

* 896 
* 11214912
* 15896912
* 20073473

### Generators and Training

For the next two questions, use the following transformation for both train and test sets:

```python
train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # ImageNet normalization
])
```

* We don't need to do any additional pre-processing for the images.
* Use `batch_size=20`
* Use `shuffle=True` for both training, but `False` for test. 

Now fit the model.

You can use this code:

```python
num_epochs = 10
history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1) # Ensure labels are float and have shape (batch_size, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        # For binary classification with BCEWithLogitsLoss, apply sigmoid to outputs before thresholding for accuracy
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_train / total_train
    history['loss'].append(epoch_loss)
    history['acc'].append(epoch_acc)

    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(validation_dataset)
    val_epoch_acc = correct_val / total_val
    history['val_loss'].append(val_epoch_loss)
    history['val_acc'].append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}"))
```

### Question 3

What is the median of training accuracy for all the epochs for this model?

* 0.05
* 0.12
* 0.40
* 0.84

### Question 4

What is the standard deviation of training loss for all the epochs for this model?

* 0.007
* 0.078
* 0.171
* 1.710


### Data Augmentation

For the next two questions, we'll generate more data using data augmentations. 

Add the following augmentations to your training data generator:

```python
transforms.RandomRotation(50),
transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
transforms.RandomHorizontalFlip(),
```

### Question 5 

Let's train our model for 10 more epochs using the same code as previously.

> **Note:** make sure you don't re-create the model.
> we want to continue training the model we already started training.

What is the mean of test loss for all the epochs for the model trained with augmentations?

* 0.008
* 0.08
* 0.88
* 8.88

### Question 6

What's the average of test accuracy for the last 5 epochs (from 6 to 10)
for the model trained with augmentations?

* 0.08
* 0.28
* 0.68
* 0.98



