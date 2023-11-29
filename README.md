# Diabetes-Prediction-Analysis

Machine Learning was employed to predict the probability of developing diabetes, contrasting Logistic Regression and Random Forest models for accuracy, precision, and recall (`diabetes_prediction.ipynb`).    
The Random Forest model, proving more accurate, was utilised to launch an interactive web page via Streamlit (`app.py`).  

A web link to the ***Diabetes Predictor page*** is provided further down in the README.

## Background
Diabetes is a chronic medical condition characterised by elevated levels of blood glucose (or blood sugar). This condition occurs when the body either does not produce enough insulin or cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood glucose and allows cells to use glucose for energy. There are three main types of diabetes:

1. **Type 1 Diabetes:**
   - Occurs when the immune system mistakenly attacks and destroys insulin-producing beta cells in the pancreas.
   - People with Type 1 diabetes require insulin injections or an insulin pump to manage their blood glucose levels.

2. **Type 2 Diabetes:**
   - Results from the body's inability to use insulin properly (insulin resistance) or insufficient production of insulin.
   - May be associated with lifestyle factors such as obesity, lack of physical activity, and genetic predisposition.
   
3. **Gestational Diabetes:**
   - Occurs during pregnancy. Resolves after childbirth.

Common symptoms of diabetes include increased thirst, frequent urination, unexplained weight loss, fatigue, and blurred vision. If left untreated, diabetes can lead to serious health consequences such as heart disease, stroke, kidney damage, nerve damage and vision problems.

Management of diabetes involves maintaining blood glucose levels within a target range through a combination of medication, lifestyle changes (such as healthy eating, regular physical activity, stress management) and regular glucose monitoring. Regular medical check-ups are essential for early detection of health consequences and effective management.

As of January 2022, it is estimated that approximately 10% of the world's adult population has diabetes. The prevalence of diabetes has been increasing globally and varies by region.  

## The Dataset  
The `diabetes_prediction_dataset.csv` file contains medical and demographic data of patients along with their diabetes status, whether positive or negative.  
It consists of various features such as `gender`, `age`, `body mass index (BMI)`, `hypertension`, `heart disease`, `smoking history`, `HbA1c level` and `blood glucose level`.  
There are 100,000 entries.

## Explanation of the data parameters  
#### Gender  
Females who have experienced gestational diabetes (diabetes during pregnancy) face an elevated risk of developing type 2 diabetes in the future. Furthermore, certain studies indicate a slightly higher diabetes risk in men compared to women.  

#### Age  
As individuals age, decreased physical activity, alterations in hormone levels and an increased probability of developing other health conditions may contribute to diabetes.  

#### Hypertension
Hypertension, or high blood pressure, is recognised as a risk factor for the onset of diabetes.  
Persistent elevated blood pressure can contribute to insulin resistance and impaired glucose metabolism, increasing the likelihood of developing type 2 diabetes.  

#### Heart Disease  
Certain heart conditions or treatments can affect glucose metabolism, potentially increasing the risk of diabetes.  

#### Smoking  
Smoking can contribute to insulin resistance and impair glucose metabolism. Quitting smoking can markedly diminish the likelihood of developing diabetes and its associated complications.  

#### BMI  
Body Mass Index (BMI) is a numerical measurement that assesses an individual's body weight in relation to their height. It is a commonly used tool to categorise individuals into different weight status categories.  
The BMI is calculated by dividing a person's weight in kilograms by the square of their height in metres.  
The formula is:  

BMI = Weight(kg) / Height(m<sup>2</sup>)  

The World Health Organization (WHO) provides the following BMI categories:  
Underweight: BMI less than 18.5  
Normal weight: BMI 18.5 to 24.9  
Overweight: BMI 25 to 29.9  
Obesity (Class I): BMI 30 to 34.9  
Obesity (Class II): BMI 35 to 39.9  
Obesity (Class III): BMI 40 or greater  

While BMI is a convenient screening tool, it does not directly measure body fat or account for variations in muscle mass, distribution of fat, and other factors.  
It is important to interpret BMI results alongside other health assessments for a more comprehensive understanding of an individual's health status.  

#### HbA1c  
HbA1c, or haemoglobin A1c, is a blood test that provides information about a person's average blood glucose levels over the past three months. It measures the percentage of haemoglobin (a protein in red blood cells) that is bound to glucose. The test is commonly used to assess long-term blood glucose management in individuals with diabetes. A higher HbA1c level indicates a higher glucose range and an increased risk of diabetes-related health consequences. The target HbA1c level varies for individuals and healthcare providers use the results to adjust treatment plans and monitor the effectiveness of diabetes management.  
For people without diabetes: An HbA1c level is typically below 5.7%.  
For people with diabetes:  An HbA1c level below 7% is often recommended.  
For some people, healthcare providers may recommend higher or lower targets based on factors such as age, overall health and the presence of other medical conditions.

#### Blood Glucose  
In diabetes, blood glucose levels are elevated due to the body's inability to effectively regulate insulin, a hormone that helps cells absorb and use glucose. There are two main types of diabetes: Type 1, where the body doesn't produce insulin, and Type 2, where the body doesn't use insulin properly. Elevated blood glucose levels can lead to various health consequences. Regular monitoring, lifestyle changes, stress management and medications are common approaches to manage blood glucose levels.   

In Australia, when individuals with diabetes monitor their blood glucose using a glucose meter, the results are typically displayed in mmol/L.  
In the United States, blood glucose levels are commonly measured in milligrams per deciliter (mg/dL).  
To convert mg/dL to mmol/L divide the number by 18.1  

Click the link below to launch the interactive web page:  
[Diabetes Predictor page](https://diabetes-prediction-analysis-2023.streamlit.app/) 

## Exploring the Data  
1. Check for missing data

<code><class 'pandas.core.frame.DataFrame'>
      RangeIndex: 100000 entries, 0 to 99999
      Data columns (total 9 columns):
       #   Column               Non-Null Count   Dtype  
      ---  ------               --------------   -----  
       0   gender               100000 non-null  object 
       1   age                  100000 non-null  float64
       2   hypertension         100000 non-null  int64  
       3   heart_disease        100000 non-null  int64  
       4   smoking_history      100000 non-null  object 
       5   bmi                  100000 non-null  float64
       6   HbA1c_level          100000 non-null  float64
       7   blood_glucose_level  100000 non-null  int64  
       8   diabetes             100000 non-null  int64  
      dtypes: float64(3), int64(4), object(2)</code>

2. Drop duplicated rows

<code>diab_pred_df.duplicated().sum()  
      3854</code>  

3. Convert `gender` to integers and drop `Other` as there were only 18 rows (100,000 rows in dataset) so should not affect the outcome.

<code>Female    56161  
      Male      39967  
      Other        18  
      Name: gender, dtype: int64</code>
    
4. Rename and regroup the `smoking_history` categories and convert to integers.

<code>df1 = df1.replace({'No Info':0, 'never':1, 'former':2, 'current':2, 'not current':2, 'ever':2})</code>

5. Visualise the data by creating a plot of each feature.

<code>p = df1.hist(figsize = (20,20))</code>

6. Find the correlation between each feature and diabetes outcome.

<code>gender                -0.04
      smoking_history        0.12
      heart_disease          0.17
      hypertension           0.20
      bmi                    0.21
      age                    0.26
      HbA1c_level            0.41
      blood_glucose_level    0.42
      diabetes               1.00
      Name: diabetes, dtype: float64</code>  
      
Blood glucose level has the highest correlation to diabetes and gender has the lowest.  

## Model Analysis
### Logistic Regression  
```
              precision    recall  f1-score   support

           0       0.96      0.99      0.98     21912
           1       0.83      0.63      0.71      2120

    accuracy                           0.96     24032
   macro avg       0.90      0.81      0.85     24032
weighted avg       0.95      0.96      0.95     24032
```

### Random Forest model  
```
              precision    recall  f1-score   support

           0       0.97      1.00      0.98     21912
           1       0.94      0.69      0.79      2120

    accuracy                           0.97     24032
   macro avg       0.96      0.84      0.89     24032
weighted avg       0.97      0.97      0.97     24032
```

**Question:**  
How well does the logistic regression model predict both the `0` (healthy participants) and `1` (participants with diabetes) labels?  
How does Random Forest model compare with logistic regression model?  

**Answer:**  
Precision: The ability of the classifier not to label as positive a sample that is negative. For class `0`, it is 96% indicating that when the model predicts class `0`(no diabetes), it is usually correct.  For class `1`, the precision is 85%, suggesting that 85% of the instances predicted as `1`(diabetes) were actually `1`.

Recall: The ability of the classifier to find all the positive samples. For class `0`, it is 99%, suggesting that the model captures most of the actual class `0` instances. For class `1`, it is lower at 63%, indicating that the model misses some of the actual class `1`(diabetes) instances.

F1-score: The harmonic mean of precision and recall. It provides a balance between precision and recall. Class `0` has a high F1-score (0.98), while class `1` has a lower but still reasonable F1-score (0.72).

Support: The number of actual occurrences of the class in this dataset. Class `0` has significantly more instances (21912) compared to class `1` (2120).

Accuracy: The overall correctness of the model, meaning that 96% of all instances are correctly classified.

Macro avg: The average of the precision, recall, and F1-score for both classes, without considering class imbalance.

Weighted avg: The weighted average of precision, recall, and F1-score, where the weights are the support values for each class. It accounts for class imbalance. Since class `0` has significantly more instances, the weighted average leans more towards the metrics for class `0`(no diabetes). The weighted average precision, recall, and F1-score are 95%, 96%, and 95% respectively.

In summary, the model performs well overall with high accuracy, but there is a notable difference in performance between the two classes. Class `0`(no diabetes) is well-predicted with high precision and recall. For class `1`(diabetes) it is weaker, especially in recall at 63%, indicating that it struggles more to correctly identify instances of class `1`.

The Random Forest model returns a higher F1-score especially for Class `1` of 79% compared to 71% for the logistic regression model. The Random Forest model also gives a better Class `1` precision of 94% and recall of 69% which means the Random Forest model is more accurate.  

We used the Random Forest model for our Diabetes Predictor web page.
 




   
  










   


   
    

     
    
    


