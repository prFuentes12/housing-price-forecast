# House Price Prediction Analysis

## 📌 Objective
The goal of this project is to develop and compare different models for predicting house prices using various regression techniques. The dataset used contains features related to real estate properties, such as square footage, number of bathrooms, location, and other relevant attributes.

## 📊 Methodology
1. **Data Preprocessing:**
   - Checked for missing values and duplicates.
   - Removed entries with zero prices.
   - Converted date columns and extracted meaningful information.
   - Encoded categorical variables using **target encoding** and **frequency encoding**.
   - Scaled numerical features where necessary.
   
2. **Feature Selection:**
   - Computed correlation matrices to identify the most relevant features for price prediction.
   - Used **Variance Inflation Factor (VIF)** to avoid multicollinearity.

3. **Model Training and Evaluation:**
   - Implemented **Simple Linear Regression** (using `sqft_living` and `bathrooms` separately).
   - Implemented **Multiple Linear Regression** with selected features.
   - Implemented **Random Forest Regressor** with all available features.
   - Evaluated models using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R² Score**.
   - Applied **Cross-Validation (5-Fold)** for robust performance evaluation.

## 📈 Results
### **Linear Regression (Simple and Multiple)**
- **Simple Linear Regression (sqft_living):**
  - **MAE:** 178,747.01
  - **RMSE:** 517.52
  - **R²:** 0.5179
  - **Cross-Validation R²:** 0.4085

- **Simple Linear Regression (bathrooms):**
  - **MAE:** 250,114.61
  - **RMSE:** 560.70
  - **R²:** 0.3577
  - **Cross-Validation R²:** 0.2206

- **Multiple Linear Regression:**
  - **MAE:** 5,447.79
  - **RMSE:** 143.29
  - **R²:** 0.9996
  - **Cross-Validation R²:** 0.9909

### **Random Forest Regression**
- **Random Forest Model:**
  - **MAE:** 9,571.01
  - **RMSE:** 383.71
  - **R²:** 0.9428
  - **Cross-Validation R²:** 0.8059

## 🔍 Conclusions
- **Multiple Linear Regression performed the best**, achieving the highest R² score and lowest error metrics. The cross-validation R² confirms its robustness.
- **Random Forest also provided strong results** but had a slightly lower R² compared to Multiple Linear Regression.
- **Simple Linear Regression models (sqft_living and bathrooms) were less effective** at predicting house prices, as they rely on a single feature.
- **Feature encoding and proper selection played a significant role** in improving model performance.
- **Cross-validation highlighted that while Multiple Linear Regression had high accuracy, Random Forest showed more variance.**

## 🚀 Future Improvements
- Hyperparameter tuning for Random Forest to improve performance.
- Explore additional features such as neighborhood data, economic indicators, and other real estate metrics.
- Implement other advanced models like Gradient Boosting or Neural Networks.


---
🔹 **Author:** Fernando  
🔹 **Contact:** fdofuentescarrasco@gmail.com  

