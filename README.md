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


<img src="https://files.slack.com/files-pri/T04USTCA709-F067GS6DF45/image.png" width="200" height="200">


<img src="https://user-images.githubusercontent.com/16319829/81180309-2b51f000-8fee-11ea-8a78-ddfe8c3412a7.png" width="150" height="280">





<class 'pandas.core.frame.DataFrame'>
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
 7   blod_glucose_level  100000 non-null  int64  
 8   diabetes             100000 non-null  int64  
dtypes: float64(3), int64(4), object(2)
memory usage: 6.9+ MB

3. Drop duplicated rows
>    diab_pred_df.duplicated().sum()
>    3854  

4. Convert `gender` to integers and drop `Other` as there were only 18 rows (100,000 rows in dataset) so should not affect the outcome.
    ```
    Female    56161
    Male      39967
    Other        18
    Name: gender, dtype: int64
    ```
5. Rename and regroup the `smoking_history` categories and convert to integers.
    ```
    df1 = df1.replace({'No Info':0, 'never':1, 'former':2, 'current':2, 'not current':2, 'ever':2})
    ```
6. Visualise the data by creating a plot of each feature.
    ```
    p = df1.hist(figsize = (20,20))
    ```
7. Find the correlation between each feature and diabetes outcome.
   ```
   gender                -0.04
   smoking_history        0.12
   heart_disease          0.17
   hypertension           0.20
   bmi                    0.21
   age                    0.26
   HbA1c_level            0.41
   blood_glucose_level    0.42
   diabetes               1.00
   Name: diabetes, dtype: float64
   ```
   Blood glucose level has the highest correlation to diabetes and gender has the lowest.


   


   
    

     
    
    


