import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.svm import SVC
import warnings


warnings.filterwarnings("ignore")

#Read in the csvs
df2022 = pd.read_csv("df2022.csv")
df2023 = pd.read_csv("df2023.csv")


#Drop the rows which have na values in variables that our model uses
df2023 = df2023.dropna(subset=[ "distanceFromMiddleScaled",
    "count_0-0", "count_0-1", "count_0-2",
    "count_1-0", "count_1-1", "count_1-2",
    "count_2-0", "count_2-1", "count_2-2",
    "count_3-0", "count_3-1", "count_3-2", "pitch_type_CH", "pitch_type_CS", "pitch_type_CU", "pitch_type_EP", 
    "pitch_type_FA", "pitch_type_FC", "pitch_type_FF", "pitch_type_FS", "pitch_type_KC", "pitch_type_PO", "pitch_type_SI", "pitch_type_SL", "pitch_type_ST",
    "pitch_type_SV"])


#Select our x variables for our training dataset
df2022X = df2022[[
    "distanceFromMiddleScaled",
    "count_0-0", "count_0-1", "count_0-2",
    "count_1-0", "count_1-1", "count_1-2",
    "count_2-0", "count_2-1", "count_2-2",
    "count_3-0", "count_3-1", "count_3-2", "pitch_type_CH", "pitch_type_CS", "pitch_type_CU", "pitch_type_EP", 
    "pitch_type_FA", "pitch_type_FC", "pitch_type_FF", "pitch_type_FS", "pitch_type_KC", "pitch_type_PO", "pitch_type_SI", "pitch_type_SL", "pitch_type_ST",
    "pitch_type_SV"]]

#Select our y variable
df2022Y = df2022["didSwing"]

#Select our x variables for the year 3 dataset
df2023X = df2023[[  
    "distanceFromMiddleScaled",
    "count_0-0", "count_0-1", "count_0-2",
    "count_1-0", "count_1-1", "count_1-2",
    "count_2-0", "count_2-1", "count_2-2",
    "count_3-0", "count_3-1", "count_3-2", "pitch_type_CH", "pitch_type_CS", "pitch_type_CU", "pitch_type_EP", 
    "pitch_type_FA", "pitch_type_FC", "pitch_type_FF", "pitch_type_FS", "pitch_type_KC", "pitch_type_PO", "pitch_type_SI", "pitch_type_SL", "pitch_type_ST",
    "pitch_type_SV"]]

model = LogisticRegression(solver = 'lbfgs')

model.fit(df2022X, df2022Y) #Fit the model
print(model.coef_) #Print the coefficients for the model


predictions = model.predict_proba(df2023X)[:,1] #Get the predictions for year 3


df2023["SwingProbability"] = predictions


df2023Original = pd.read_csv("df2023.csv")

#Add the predictions to the original year3.csv file dataframe
df2023WithPredictions = pd.merge(df2023Original, df2023[['release_speed', 'batter','SwingProbability','pfx_x','pfx_z']], on=['batter','release_speed','pfx_x','pfx_z'], how='left')

#This for loop adds in predictions for rows where the prediction is missing by using a simplified model that only considers the count
for i in range(len(df2023WithPredictions)): #Go through the year3 dataframe that has the predictions
    if pd.isna(df2023WithPredictions["SwingProbability"][i]): #If there is no prediction (this means an input variable was missing so the model couldn't predict)
        #Then, we go through all counts and assign a swing probability according to historical swing probability by counts from years 1 and 2
        if df2023WithPredictions["balls"][i] == 3 and df2023WithPredictions["strikes"][i] == 0:
            df2023WithPredictions["SwingProbability"][i] = .108396
        elif df2023WithPredictions["balls"][i] == 0 and df2023WithPredictions["strikes"][i] == 0:
            df2023WithPredictions["SwingProbability"][i] = .306867
        elif df2023WithPredictions["balls"][i] == 1 and df2023WithPredictions["strikes"][i] == 0:
            df2023WithPredictions["SwingProbability"][i] = 0.427487
        elif df2023WithPredictions["balls"][i] == 2 and df2023WithPredictions["strikes"][i] == 0:
            df2023WithPredictions["SwingProbability"][i] = 0.428494
        elif df2023WithPredictions["balls"][i] == 0 and df2023WithPredictions["strikes"][i] == 1:
            df2023WithPredictions["SwingProbability"][i] = 0.491472
        elif df2023WithPredictions["balls"][i] == 0 and df2023WithPredictions["strikes"][i] == 2:
            df2023WithPredictions["SwingProbability"][i] = 0.517369
        elif df2023WithPredictions["balls"][i] == 1 and df2023WithPredictions["strikes"][i] == 1:
            df2023WithPredictions["SwingProbability"][i] = 0.541855
        elif df2023WithPredictions["balls"][i] == 3 and df2023WithPredictions["strikes"][i] == 1:
            df2023WithPredictions["SwingProbability"][i] = 0.547043
        elif df2023WithPredictions["balls"][i] == 1 and df2023WithPredictions["strikes"][i] == 2:
            df2023WithPredictions["SwingProbability"][i] = 0.578408
        elif df2023WithPredictions["balls"][i] == 2 and df2023WithPredictions["strikes"][i] == 1:
            df2023WithPredictions["SwingProbability"][i] = 0.579297
        elif df2023WithPredictions["balls"][i] == 2 and df2023WithPredictions["strikes"][i] == 2:
            df2023WithPredictions["SwingProbability"][i] = 0.651132
        elif df2023WithPredictions["balls"][i] == 3 and df2023WithPredictions["strikes"][i] == 2:
            df2023WithPredictions["SwingProbability"][i] = 0.712231
        else:
            df2023WithPredictions["SwingProbability"][i] = .47675755
        

df2023WithPredictions.to_csv("df2023WithPredictions.csv", index=False) #output our year3 + predictions as to a validation.csv file


