-----------

Title: Performance analysis of  Machine Learning Models trained on life threatening Diseases<br>
Date: "November 2021"<br>
Author: Waqas Kureshy, Neeharika Yeluri, Jasmine Wang, San José State University<br>

-------------------


# Abstract

Humans are plagued by different diseases globally, of these diseases CVD (Cardiovascular diseases), Diabetes and Breast Cancer are known to have a prolonged and detrimental  effect. Our task was to work on datasets related to the diseases mentioned above by employing different Machine Learning approaches, consolidate results and provide a comparison to the reader about the models used. For this exercise we used Sequential models, Decision tree Classifiers, Support Vector Machines, LogisticRegression, RandomForestClassifier, and KNeighborsClassifier. Techniques used and comparison for all the models made for each dataset in terms of metrics are presented in this paper.

# Introduction

Cardiovascular diseases, diabetes, and breast cancer are top 3 causes of death globally, taking millions of lives each year, which accounts for 50% of all deaths worldwide. It has long been the struggle of Researchers, Scientists and Doctors to come up with comprehensive statistical models to conclude positively the patient's medical ailments. This approach carries benefits for the Medical industry workers as well as the patient. Employing such models that have the ability to classify a patient’s health could serve as an early warning system, this enables the patient to get timely medical treatment and advice. This approach as well as proving valuable to a person's health can also potentially save time and effort  in the process of Medical diagnosis. Our objective for this project was to take three datasets pertaining to Cardiovascular diseases, Diabetes, as well as Breast Cancer, conduct initial exploratory analysis of the data, clean and preprocess the data and then to use this data to make different Machine Learning models, compile metrics and present an analysis of the performance of each model to the reader.


For this exercise we worked on three different datasets, feature and statistics of each dataset are elaborated as:

## Cardiovascular Disease Classification Dataset

| Feature     | Data type     | value |
| :-------------: | :-------------: |:-------------:|
|Age            |       Integer    | Age of the patient  [years] |
| Sex | Integer |    Sex of the patient [1: Male, 0: Female] |
| Chest Pain Type | Integer | [1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal Pain,4: Asymptomatic] |
| Resting Blood Pressure | Integer| Resting blood pressure [mm Hg] |
| Cholesterol | Integer| Serum cholesterol [mm/dl] |
| Fasting Blood Sugar| Integer | [1: if FastingBS > 120 mg/dl, 0: otherwise] |
| Resting ECG | Integer | Resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2: showing probable or definite left ventricular hypertrophy by Estes criteria]|
| Max Heart Rate | Integer| Maximum heart rate achieved [Numeric value between 60 and 202] |
| Exercise Angina | Integer | Exercise-induced angina [1: Yes, 0: No] |
| Old peak | Float | Oldpeak = ST [Numeric value measured in depression] |
| ST Slope | Integer | The slope of the peak exercise ST segment [1: upsloping,  2: flat, 3: downsloping] |
| Target | Integer | Output class [1: heart disease, 0: Normal] |

-----

The Correlation of the dataset attributes with respect to the target variable i.e the possibility of having a CVD (Cardiovascular Disease) are depicted in the graph number below.<br>
![image](https://user-images.githubusercontent.com/78277453/204978541-16bef032-5451-4fdc-b1df-927cd0e2b3aa.png)



From this analysis it is evident that 9 different attributes/features have a positive correlation with the target variable.<br>

A heatmap showing the correlation of the features with each other are shown in fig-1 , this image shows graphically how each feature correlates with each other feature. This analysis shows that the following features are positively correlated with the target variable [Age , Sex , Chest pain Type ,  Resting Blood Pressure, Fasting Blood Sugar ,  Resting ECG ,  Exercise Angina , Old Peak , ST_Slope ]<br>

![image](https://user-images.githubusercontent.com/78277453/204979205-70461ff2-5f1a-4c46-b7a6-cc50636c3ee1.png)

With the attribute ST Slope (The ST segment encompasses the region between the end of ventricular depolarization and beginning of ventricular repolarization on the ECG ) showing the highest correlation.<br>

The CVD dataset is composed of 1190 cases which is professionally compiled, by merging datasets from different origins and compilations presented in [1] .The distribution of the dataset with respect to the target variable as shown in fig-2 shows that the dataset is balanced and is adequate for modeling. <br>
![image](https://user-images.githubusercontent.com/78277453/204979389-2d6bfd91-ff16-423f-b42c-1dbe9408450e.png)


A graph plot shown in fig-3 shows the distribution of the dataset with respect to the target variable differentiated on the basis of age (0: for no CVD disease and 1: for having a positive identification of CVD).<br>
![image](https://user-images.githubusercontent.com/78277453/204979557-03dcb5bd-ca07-4f62-b752-1e7cfa6680d7.png)


The following fig-4 shows a  plot of  various types of chest pain encountered in the dataset separated by Gender, where 0 represents female cases and 1 represents male cases.<br>
![image](https://user-images.githubusercontent.com/78277453/204979666-66957bed-bd98-4479-be46-74f2d4cc04a0.png)


The following fig-5 shows a  plot of the distribution of Age across the Sex feature, it can be observed that males tend to contract Heart Diseases earlier than females. In the figure 0 represents female cases and 1 represents male cases.<br>
![image](https://user-images.githubusercontent.com/78277453/204979958-47a643df-0002-4665-b1a3-a3c8f1d895b2.png)


The plot in fig-6 shows interesting information about the trends of heart rate of people having CVD and not having CVD separated by Gender. It can be observed that Males who have suffered from CVD tend to have a lower Heart Beat rate than their female counterparts who also have CVDs.<br>
![image](https://user-images.githubusercontent.com/78277453/204980080-a400ba38-4742-4917-b225-09ad3fe46a44.png)

## Diabetes
Diabetes is a common chronic disease. Prediction of diabetes at an early stage can lead to improved treatment. The dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.
Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. This dataset has 9 attributes which are listed below:

Pregnancies		    :	Number of times pregnant
Glucose 			    :	Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure		    :	Diastolic blood pressure (mm Hg)
SkinThickness		    :	Triceps skin fold thickness (mm)
Insulin			    :	2-Hour serum insulin (mu U/ml)
BMI	 			    :	Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction : 	Diabetes pedigree function
Age	 				Age (years)
Outcome			    :	Class variable (0: Tested Negative or 1: Tested Positive)
There are only numerical variables in this dataset. 768 observations, and 9 variables(1 dependent) are available

**Imputation:**
Under normal circumstances, it seems that there are no missing values in the data set, but there may be missing values hidden in the data of the variables here.
Then we examined the missing values of each variable according to the target variable. So we decided to apply different methods in order to fill na values according to the state of each variable because of the range differences of flag counts.<br>
For the variables {Glucose, Blood Pressure and BMI} we filled the missing values with median and for the remaining two variables {Insulin and SkinThickness} we filled them with the K Nearest Neighbours.

**New Feature Interaction:**

By converting these numerical variables into categorical we get a clear picture of the analysis of data.
- Glucose - Women with hyperglycemia will have a higher incidence of diabetes on average the "Outcome".
- Age - Middle-aged women will have a higher incidence of diabetes on average the "Outcome".
- BMI - Morbidly obese women will have a higher incidence of diabetes on average the "Outcome".
- Blood Pressure - Women with high blood pressure will have a higher incidence of diabetes on average the "Outcome".
- Insulin - Women with abnormal insulin will have a higher incidence of diabetes on average the "Outcome."
- Pregnancies - Women with a very high pregnancy rate will have a higher incidence of diabetes on average.

<br>

**Correlation:** <br>
All the attributes have a positive correlation with respect to the target variable Outcome which is depicted in the table below with Glucose having the highest correlation and BloodPressure the lowest.

![image](https://user-images.githubusercontent.com/78277453/205763504-ccfebfbc-e5fe-431c-bbf0-05c78f833ea8.png)

In the case of diabetes, especially the Glucose, BMI and Age variables of women are an important factor. The rate of diabetes may be higher in middle-aged women aged 45-65 years.
<br>


**Heat Map:** <br>
The following is the heat map which shows  the correlation of the features with each other. This image shows graphically how each feature correlates with each other feature. This analysis shows that the following features are positively correlated with the target variable [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction and Age].

![image](https://user-images.githubusercontent.com/78277453/205763611-91e8a603-52cb-4190-9316-1c8e689111d8.png)


The following pie chart  shows the number of people diagnosed and non diagnosed with diabetes.

![image](https://user-images.githubusercontent.com/78277453/205763662-8dbc7b64-2fd8-4020-b719-f5ea640f6efd.png)


Below are the distributions plots of each feature

**Glucose Distribution Plot:**

![image](https://user-images.githubusercontent.com/78277453/205763770-fd398942-ea0a-4215-99c9-bb9d19bda7b9.png)

**BloodPressure Distribution Plot:**

![image](https://user-images.githubusercontent.com/78277453/205764023-d082f130-429b-4f37-ba2a-762a5192f3b9.png)


**Insulin Distribution Plot:**

![image](https://user-images.githubusercontent.com/78277453/205764124-e0846c37-120c-4d63-9cde-60b85e551306.png)


**Pregnancies Histogram Plot:**

![image](https://user-images.githubusercontent.com/78277453/205764184-ad4e4256-8481-40f0-9558-d1b9975afde4.png)

**Age Histogram Plot:**

![image](https://user-images.githubusercontent.com/78277453/205764231-61e363b8-c1ea-4366-8982-612282de1e61.png)

**Pair plot:**

![image](https://user-images.githubusercontent.com/78277453/205764428-86d346ff-b90b-48e3-8471-41cf02518a65.png)


# Methods
Each dataset used, presents itself essentially as a classification task. For this reason we employed different  classification algorithms. The design methodology and Algorithm used for each dataset is explained further in the following sections.

## CVD (Cardiovascular Disease Classification using ANNs)
For this task a Artificial Neural Network model was trained, to classify patient cases as having the Target ( CVD ) diseases or not. The methodology is summarized as:

- 11 data columns were selected as features and the ‘Target’ column was selected as our label.
- The set was distributed into training and testing data and then scaled using a Minmax scaler.
- A sequential model was created using  the Tensorflow Keras API which comprised 3 hidden layers and 1 output layer.
- The three hidden layers used the SELU ( ) activation function, this function was used for normalization. Each layer had fewer neurons than the preceding layer.
- A sigmoid function was used in the last layer to output the probabilities for the predicted target.
- A Dropout layer  was used after every Dense layer to avoid overfitting.
- A callback EarlyStopping which monitored the validation loss metric was also used to prevent overfitting.
- The neural network was trained for 600 epochs , but the training was stopped at 85 epochs due to the early stop callback.
- The model’s metrics showing the model’s Loss, Validation-Loss,Accuracy and Validation-Accuracy are shown in the fig-7 below.

![image](https://user-images.githubusercontent.com/78277453/204981172-3f0473d3-f6c7-4eea-b023-88d9be20e2cf.png)


The last recorded metrics are shown in table below.
|loss| acuracy |val loss | val accuracy |
|-----|----|------|---------------------|
| 0.4256|0.8271|0.3684|  0.8431 |

<br>

## CVD (Cardiovascular Disease) Classification using SVM

For this task a SVD model was trained, to classify patient cases as having the Target ( CVD ) diseases or not. The methodology is summarized as:

- Exploratory data analysis was conducted using the Seaborn library the results of which are summarized and displayed graphically in the introduction section.
- 11 data columns were selected as features and the ‘Target’ column was selected as our label.
- The set was distributed into training and testing data and then scaled using a Minmax scaler.
- The model was evaluated using the Confusion matrix and Classification Report metrics. These results are summarized and compared in the Comparison section along with other algorithms.


## CVD (Cardiovascular Disease Classification) using Decision Trees

For this task a Decision Tree model was trained, to classify patient cases as having the Target ( CVD ) diseases or not. The methodology is summarized as:

- Exploratory data analysis was conducted using the Seaborn library the results of which are summarized and displayed graphically in the introduction section.
- 11 data columns were selected as features and the ‘Target’ column was selected as our label.
- The set was distributed into training and testing data and then scaled using a Minmax scaler.
- The model was trained using a Decision Tree classifier with the entropy criterion with a max depth of 4. The image of the Tree is shown in the notebook.
- The model was evaluated using the Confusion matrix and Classification Report metrics. These results are summarized and compared in the Comparison section along with other algorithms.


## Diabetes Using Logistic Regression:

For this task a Logistic regression model was trained, to classify patient cases as having the Target (Diabetes) diseases or not. The methodology is summarized as:

- 8 data columns are selected as features and the target column (Outcome) is selected as the label.
- The dataset was distributed into training and testing data and then scaled using Standard scaler.
- The model was trained using a Logistic Regression method and the accuracy was predicted.
- The model was evaluated using the Confusion matrix and Classification Report metrics. These results are summarized and compared in the Comparison section along with other algorithms.

## Diabetes Using Decision Tree Classification:

For this task a Decision Tree model was trained, to classify patient cases as having the Target (Diabetes) diseases or not. The methodology is summarized as:

- 8 data columns are selected as features and the target column (Outcome) is selected as the label.
- The model was trained using a Decision Tree classifier with the entropy criterion with a max depth of 3. The image of the Tree is shown in the notebook.
- The model was evaluated using the Accuracy, Confusion matrix and Classification Report metrics of training and testing data. These results are summarized and compared in the Comparison section along with other algorithms.


## Diabetes Using SVM:

For this task a SVC model was trained, to classify patient cases as having the Target (Diabetes) diseases or not. The methodology is summarized as:

- 8 data columns are selected as features and the target column (Outcome) is selected as the label.
- Exploratory data analysis was conducted using the Seaborn library the results of which are summarized and displayed graphically in the introduction section.
- The model was trained using a Support Vector Classifier and then Accuracy, Confusion Matrix and the Classification Report metrics were evaluated for both the training and testing data.



-------------------


# Example Analysis
Here we use the **Breast Cancer Dataset** as an example to illustrate the process.


## Define the question
Breast cancer is the most common cancer amongst women in the world. It accounts for 25% of all cancer cases, and affects millions of people each and every year. It starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.  The key challenge against its detection is how to classify tumors into malignant (cancerous) or benign(non cancerous).

Here we use Breast Cancer Wisconsin (Diagnostic) Dataset. It provides rich features describing tumors (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) from different angles(mean, standard deviation, and worst). Additionally it has preliminary diagnosis as M(alignant) or B(enign). So it’s adequate for modeling.

Based on the dataset, we think it’s perfect to complete the analysis of these tumors using machine learning various classification algorithms.


## Tidy the data
The dataset contains 569 records with 32 attributes.  These attributes’ name, non-null count, data type represented in the dataset are shown in fig.

![image](https://github.com/Kureshy/Multi-Health-Classification/blob/main/paper/Images/ds.info.png)


We eliminate the  _worst features since these extreme features don’t represent a fair distribution of records. Also we also eliminate the _se features because they are similar to the _mean features while the _mean features are better to represent the records considering the size of the dataset. Additionally, we change the diagnosis feature from categorical[M, B] to numerical[1, 0] for easy processing later. Now data showing in fig are neat and ready for further analysis.

![image](https://github.com/Kureshy/Multi-Health-Classification/blob/main/paper/Images/ds.data_ready.png)

## Explore the data

Now we have the mean features together with the diagonal feature for further analysis. We can take a closer look at these features as well as from a statistical point of view, showing in fig.

![image](https://github.com/Kureshy/Multi-Health-Classification/blob/main/paper/Images/ds.describe.png)


A heatmap showing the correlation of the features with each other is shown in fig.  The image shows graphically how each feature correlates with each other feature. This analysis shows that the following features [‘radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean] are positively correlated with the target variable[diagnosis] , with[‘radius_mean’, ‘perimeter_mean’, ‘area_mean’, ‘concave points_mean', ‘concavity_mean'] showing the highest correlation.


![image](https://github.com/Kureshy/Multi-Health-Classification/blob/main/paper/Images/ds.heatmap.png)

The above is proved by fig. showing indeed [‘radius_mean’, ‘perimeter_mean’, ‘area_mean’, ‘concave points_mean', ‘concavity_mean'] are major contributors to diagnosis.

![image](https://github.com/Kureshy/Multi-Health-Classification/blob/main/paper/Images/ds.major_contributors.png)

Furthermore we understand the distribution of Malignant vs. Benign graphed in fig.

![image](https://github.com/Kureshy/Multi-Health-Classification/blob/main/paper/Images/ds.counts.png)

Moreover we have an impression regarding both distribution of a single feature as well as its relationships with other features in fig.

![image](https://github.com/Kureshy/Multi-Health-Classification/blob/main/paper/Images/ds.pairplot.png)

## Use the models in one shot (RandomForestClassifier,  KNeighborsClassifier, GaussianNB,  SGDClassifier)
10 data columns ending with _mean are selected as features, and the diagnosis column is selected as a label.

StandardScaler is used to standardize X.

The dataset is split into training and testing data as 80% vs. 20%.

In one shot, multiple models (RandomForestClassifier, KNeighborsClassifier, GaussianNB, SGDClassifier) are used.


## Perform the analysis
In one shot again, all models are evaluated using accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix. Please see fig & fig.

![image](https://github.com/Kureshy/Multi-Health-Classification/blob/main/paper/Images/ds.metrics1.png)
![image](https://github.com/Kureshy/Multi-Health-Classification/blob/main/paper/Images/ds.metrics2.png)


## Check results
All models’ overall scores are pretty high: 93% - 96% while there’s no champion forever since data are shuffled for each and every run.

## Btw:
I test RandomForestClassifier, KNeighborsClassifier, GaussianNB and SGDClassifieruse, together with LogisticRegression and SupportVectorMachine, all of them achieve 93% - 96%. I keep the former four while drop the latter two since others use them. 

I use the former four models in one shot so that it’s code-efficient and it’s easy for model comparison. 




# Comparisons

# Conclusions
In this section we compile and compare the various Evaluation metrics of each and every model.

## CVD (Cardiovascular Disease) Classification

Three models were created for the CVD dataset, as listed below:
- CVD (Cardiovascular Disease) Classification using ANNs
- CVD (Cardiovascular Disease) Classification using SVM
- CVD (Cardiovascular Disease) Classification using Decision Trees

The image of Confusion Matrix for each model is shown as below :

![image](https://user-images.githubusercontent.com/78277453/204982208-c2d2be67-f95d-45dc-81e5-8cd19946d9a7.png)
<br>
Confusion Matrix for ANN model
<br>

![image](https://user-images.githubusercontent.com/78277453/204982402-5cd0625d-341e-4482-9ca2-1c2fce2d2a79.png)
<br>
Confusion Matrix for SVM model
<br>

![image](https://user-images.githubusercontent.com/78277453/204982504-4d22e1aa-f17d-465b-8080-84eefd723b6a.png)
<br>
Confusion Matrix for Decision Tree Classifier
<br>

For easy comparison a table is shown below that compiles all the True Positive, False Positive, True Negative and False Negative values gathered from the confusion matrix for each model trained on the CVD dataset.

| Model name | TP | FP | FN | TN |
|---------- | --- | --- | ---| ----|
| SVM       |41.46%  |9.80% |5.04% |43.70%|
| ANN      | 38.38% | 8.68% | 7% | 45.94% |
| Decision Tree | 40.06% | 7.56% | 10.36%  | 42.02% |

The classification report for each model trained on the CVD dataset is shown in the table with its respective label below.




|     | Precision|   Recall |   F-1|
|----  | --------  | --------- | ---- |
| 0    | 0.85 |     0.82    |    0.83 |
| 1    | 0.84  |    0.87  |      0.85|
|accuracy |    |          |     0.84 |

Table: Classification Report for ANN model

|    |  Precision |   Recall |   F-1 |
| ---- | --------- |  --------- |---|
|0 |    0.85 |    0.82 |     0.83 |
|1 |    0.84 |       0.87 |      0.86 |
|accuracy |             |       0.85 |

Table: Classification Report Decision Tree model

|      |Precision  | Recall  |  F-1 |
| ----  |---------  |  ---------| ---- |
|0  |  0.89 |         0.81  |    0.85 |
| 1  |   0.82 |       0.90  |      0.85 |
|accuracy |    |            |    0.85   |

Table: Classification Report for SVM model


Comparing all the models given their metrics, all the models performed somewhat similarly but out of the three models the SVM model performed the best.

# Conclusion

Learning is possible considering the two step process, for at least finite hypothesis sets. Our Datasets and hypothesis sets are fixed, the data is assumed to be IID (Independent Identically Distributed) generated from a target distribution P[y|x], from this same distribution we obtain Independent test points for each dataset with respect to which we calculate Eout. This means that when we are considering the first step of the Two Step Learning approach i.e Eout(g) nearly equal to Ein, the testing and training points come from the same distribution. Next we ensure that the same error measure is used by the algorithm for selecting the best hypothesis as well as for when comparing "g" nearly equal to "f" that is Eout.<br>

We understand that its not the question of choosing the most complicated hypothesis set, because a much smaller hypothesis set may very well be able to adapt to the problem (Optimal Tradeoff).<br>

The fundamental Learning approach dictates that for a complex Target function we would need a larger set of Hypothesis and in order to maintain Eout nearly equal to Ein we would need a  larger set of data.<br>

Our experiments are made up of complex datasets and incidentally their respective target functions are also complex, so the more the data our models has to train on would be beneficial. One major attribute of learning is the "Error", which is user defined and to measure error in our experiments we have used metrics like Classification report and Confusion matrices  to quantify the error of our trained models. This error effects how we select "g", our hypothesis.<br>

Additionally we have implemented multiple models and compiled their Error metrics so that selection of a suitable model specific to the dataset, that outperforms others can be selected.<br>


# References

[Cardiovascular Disease Prediction Dataset - OpenML](https://www.openml.org/search?type=data&status=active&id=43672) <br>
[Breast Cancer Prediction Dataset - OpenML](https://www.openml.org/search?type=data&status=active&id=1510)<br>
[Diabetes Dataset - OpenML](https://www.openml.org/search?type=data&status=active&id=42608)
