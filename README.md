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

$${\color{teal}Click \space \color{teal}the \space \color{teal}link \space \color{teal}below \space \color{teal}to \space \color{teal}launch \space \color{teal}the \space \color{teal}interactive \space \color{teal}web \space \color{teal}page:}$$  
$${\color{red}Welcome \space \color{lightblue}To \space \color{lightgreen}Github}$$ 
[Diabetes Predictor page](https://diabetes-prediction-analysis-2023.streamlit.app/) 

## Exploring the Data  
1. Check for missing data

   <img src="https://github.com/EvaB5050/Diabetes-Prediction-Analysis/assets/131424690/d5357f1d-71f1-4cbc-a6fa-bdb9052af65b" width="300" height="200">


2. Drop duplicated rows

   <img src="https://github.com/EvaB5050/Diabetes-Prediction-Analysis/assets/131424690/e2b3da85-73dd-45a6-ad35-d1972a11b0b1" width="300" height="80">


3. Convert `gender` to integers and drop `Other` as there were only 18 rows (100,000 rows in dataset) so should not affect the outcome.

   <img src="https://github.com/EvaB5050/Diabetes-Prediction-Analysis/assets/131424690/0aee639c-f299-4f59-91bc-cafda90b837c" width="300" height="120">

    
4. Rename and regroup the `smoking_history` categories and convert to integers.

   <img src="https://github.com/EvaB5050/Diabetes-Prediction-Analysis/assets/131424690/56702a39-5c7f-4f30-8e4b-f95cff77ea6a" width="900" height="35">


5. Visualise the data by creating a plot of each feature.

   <img src="https://github.com/EvaB5050/Diabetes-Prediction-Analysis/assets/131424690/29380a7c-f552-4d2c-8a76-813b4e56f670" width="300" height="35">


6. Find the correlation between each feature and diabetes outcome.

   <img src="https://github.com/EvaB5050/Diabetes-Prediction-Analysis/assets/131424690/bc308e72-49d0-47d9-9cca-80b2ec927fa7" width="300" height="200">

      
Blood glucose level has the highest correlation to diabetes and gender has the lowest.  

---
## Model Analysis
### Logistic Regression  
<img src="https://github.com/EvaB5050/Diabetes-Prediction-Analysis/assets/131424690/8b66a311-9441-4344-b31a-9489fa016a9d" width="500" height="200">

### Random Forest model  
<img src="https://github.com/EvaB5050/Diabetes-Prediction-Analysis/assets/131424690/a837096b-5c02-4807-991e-344fe0956fe8" width="500" height="200">  


## Comparison:

Class 0 (Healthy Participants):  
Both models perform well for classifying healthy participants (label 0).
The Random Forest model has a slightly higher precision (97% vs. 96%) and perfect recall (100% vs. 99%) for this class.
F1-scores are comparable between the two models for class 0.  

Class 1 (Participants with Diabetes):  
The Random Forest model outperforms the Logistic Regression model for classifying participants with diabetes (label 1).
The Random Forest model has a higher precision (94% vs. 83%) and recall (69% vs. 63%).
The f1-score for class 1 is notably higher for the Random Forest model (79% vs. 71%).  

In summary, both models excel in classifying healthy participants (label 0), with the Random Forest model showing a slight edge. However, the Random Forest model significantly outperforms the Logistic Regression model in identifying participants with diabetes (label 1), achieving higher precision, recall, and f1-score. This suggests that the Random Forest model is more effective in distinguishing individuals with diabetes from the healthy population compared to the Logistic Regression model.

We used the Random Forest model for our Diabetes Predictor web page.
 




   
  










   


   
    

     
    
    


