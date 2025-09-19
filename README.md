


# Homework 1



---

## Getting the Data
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
