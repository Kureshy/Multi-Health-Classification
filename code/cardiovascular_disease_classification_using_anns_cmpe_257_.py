# -*- coding: utf-8 -*-
"""Cardiovascular Disease Classification using ANNs CMPE-257 .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ocqoinuERASze-0R8s3SQ3epRznvtOiZ

# About the dataset

The aim of the model of this notebook is to create a heart failure predictor, the dataset used is an amalgamation of five different datasets that have a conjunction on eleven different features.

# Attributes
- **Age**: age of the patient [years]
- **Sex**: sex of the patient [1: Male, 0: Female]
- **ChestPainType**: chest pain type [1: Typical Angina , 2:   Atypical Angina, 3: Non-Anginal Pain, 4: Asymptomatic]
- **Resting_BP**: resting blood pressure [mm Hg]
- **Cholesterol**: serum cholesterol [mm/dl]
- **Fasting_Blood_Sugar**: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- **Resting_ECG**: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- **Max_Heart_Rate**: maximum heart rate achieved [Numeric value between 60 and 202]
- **Exercise_Angina**: exercise-induced angina [1: Yes, 0: No]
- **Old_Peak**: oldpeak = ST [Numeric value measured in depression]
- **ST_Slope**: the slope of the peak exercise ST segment [1: upsloping, 2: flat, 3: downsloping]
- **Target**: output class [1: heart disease, 0: Normal]

--------

Creating a Lookup table for the attributes
"""

# A simple dictionary for looking up attributes
attribute_lookup={'Age': 'age of the patient [years]','Sex': 'sex of the patient [1: Male, 0: Female]',
'Chest_pain_Type': 'chest pain type [1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal Pain, 4: Asymptomatic]',
'Resting_BP': 'resting blood pressure [mm Hg]',
'Cholesterol': 'serum cholesterol [mm/dl]',
'Fasting_Blood_Sugar': 'fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]',
'Resting_ECG': 'resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2: showing probable or definite left ventricular hypertrophy by Estes criteria]',
'Max_Heart_Rate': 'maximum heart rate achieved [Numeric value between 60 and 202]',
'Exercise_Angina':'exercise-induced angina [1: Yes, 0: No]',
'Old_Peak': 'oldpeak = ST [Numeric value measured in depression]',
'ST_Slope': 'the slope of the peak exercise ST segment [1: upsloping, 2: flat, 3: downsloping]',
'Target': 'output class [1: heart disease, 0: Normal]'}

#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""**Mounting Google Drive**"""

pwd

from google.colab import drive
drive.mount('/content/gdrive')

!ls gdrive/MyDrive/Cmpe-257

!cp gdrive/MyDrive/Cmpe-257/'CVD dataset.csv' ./

df=pd.read_csv('CVD dataset.csv')

"""# Exploratory Data analysis"""

df.head()

df.shape

df.describe().transpose()

df.corr()['Target'].sort_values()

"""**Correlation with respect to the Target**"""

df.corr()['Target'].sort_values().plot(kind='bar')

"""**Correlation Matrix**"""

plt.figure(figsize=(12,15))
sns.heatmap(df.corr(),annot=True)

df.info()

sns.countplot(x='Target',data=df)

plt.figure(figsize=(15,10))
sns.countplot(df['Age'],hue=df['Target'])

attribute_lookup

sns.countplot(x=df['Target'],data=df,hue='Sex')

"""**The following plot shows various types of chest pain encountered in the dataset sepearted by Gender, where 0 represents female cases and 1 represents male cases.**"""

sns.countplot(x='Chest_pain_type',data=df,hue='Sex')

"""**The following plot shows the distribution of Age across the Sex feature**"""

sns.boxplot(x='Sex',y='Age',data=df)

"""**The plot in the figure below shows interesting information about the trends of heart rate of people having CVD and not having CVD separated by Gender. It can be observed that Males who have suffered from CVD tend to have a lower Heart Beat rate than their female counterparts who also have CVDs.**"""

sns.boxplot(x='Target',y='Max_Heart_Rate',data=df,hue='Sex')

"""**Scatter plot of Heart Rate Vs. Resting BP sepaerated by Age and Gender**"""

plt.figure(figsize=(12,10))
sns.scatterplot(x='Max_Heart_Rate',y='Resting_BP',data=df,hue='Sex',size='Age')

sns.pairplot(df)

"""# Model"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense,Flatten,InputLayer
from tensorflow.keras import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X=df.drop('Target',axis=1).values

y=df['Target'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

scaler=MinMaxScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

from tensorflow.keras.layers import Dropout

early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=20)

model=Sequential()
#model.add(InputLayer(input_shape=(11,)))
model.add(Dense(30,activation='selu'))
model.add(Dropout(0.5))
model.add(Dense(15,activation='selu'))
model.add(Dropout(0.5))
model.add(Dense(8,activation='selu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),callbacks=[early_stop])

"""# Evaluating Metrics"""

model_performence_metrics=pd.DataFrame(model.history.history)
model_performence_metrics.plot()

predictions=(model.predict(X_test) > 0.5).astype("int32")

test_predictions=pd.Series(predictions.reshape(357,))

pred_df=pd.DataFrame(y_test,columns=['Test True Y'])

pred_df=pd.concat([pred_df,test_predictions],axis=1)
pred_df.columns=['Test True Y','Model_predictions']
pred_df

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,predictions))

cf_mat=confusion_matrix(y_test,predictions)
print(sns.heatmap(cf_mat/np.sum(cf_mat), annot=True, fmt='.2%'))

print(classification_report(y_test,predictions))

model.save('CVD_classification.h5')