import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.svm import SVC
import warnings
from sklearn.model_selection import train_test_split

#Read in the csvs
df2022 = pd.read_csv("df2022.csv")
df2023 = pd.read_csv("df2023.csv")

df = pd.concat([df2022, df2023], ignore_index=True) 


X = df[["distanceFromMiddleScaled"]]  
y = df['didSwing']           

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#DISTANCE ONLY MODEL

modelDistanceOnly = LogisticRegression(solver = 'lbfgs')

modelDistanceOnly.fit(X_train, y_train)
score1 = modelDistanceOnly.score(X_test, y_test)



# DISTANCE AND COUNT MODEL

X = df[["distanceFromMiddleScaled",
    "count_0-0", "count_0-1", "count_0-2",
    "count_1-0", "count_1-1", "count_1-2",
    "count_2-0", "count_2-1", "count_2-2",
    "count_3-0", "count_3-1", "count_3-2"]]  
y = df['didSwing']     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


modelDistanceAndCount = LogisticRegression(solver = 'lbfgs')

modelDistanceAndCount.fit(X_train, y_train)
score2 = modelDistanceAndCount.score(X_test, y_test)



#DISTANCE, COUNT, AND PITCH TYPE MODEL

X = df[["distanceFromMiddleScaled",
    "count_0-0", "count_0-1", "count_0-2",
    "count_1-0", "count_1-1", "count_1-2",
    "count_2-0", "count_2-1", "count_2-2",
    "count_3-0", "count_3-1", "count_3-2", "pitch_type_CH", "pitch_type_CS", "pitch_type_CU", "pitch_type_EP", 
    "pitch_type_FA", "pitch_type_FC", "pitch_type_FF", "pitch_type_FS", "pitch_type_KC", 
    "pitch_type_KN", "pitch_type_PO", "pitch_type_SI", "pitch_type_SL", "pitch_type_ST",
    "pitch_type_SV"]]  
y = df['didSwing']     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelDistanceCountPitchType = LogisticRegression(solver = 'lbfgs')

modelDistanceCountPitchType.fit(X_train, y_train)

score3 = modelDistanceCountPitchType.score(X_test, y_test)



#DISTANCE, COUNT, AND PITCH CLASSIFICATION MODEL

X = df[["distanceFromMiddleScaled",
    "count_0-0", "count_0-1", "count_0-2",
    "count_1-0", "count_1-1", "count_1-2",
    "count_2-0", "count_2-1", "count_2-2",
    "count_3-0", "count_3-1", "count_3-2","pitchClassification_Fastball","pitchClassification_Offspeed","pitchClassification_Breaking Ball"]]  
y = df['didSwing']     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelDistanceCountPitchClassification = LogisticRegression(solver = 'lbfgs')

modelDistanceCountPitchClassification.fit(X_train, y_train)

score4 = modelDistanceCountPitchClassification.score(X_test, y_test)

#Print the scores from these models
print("Distance Only Model:" + str(score1))

print("Distance and Count Model:" + str(score2))

print("Distance, Count, and Pitch Type Model:" + str(score3))

print("Distance, Count, and Pitch Classification Model:" + str(score4))


