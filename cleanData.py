import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.svm import SVC
import warnings
from pybaseball import statcast



df2022 = statcast(start_dt='2022-04-07', end_dt='2023-03-07')
df2022 = df2022[['release_speed','batter','pitcher','description','stand','p_throws','pitch_type','balls','strikes','pfx_x','pfx_z','plate_x','plate_z','sz_top','sz_bot']]
df2022 = df2022.reset_index()

df2023 = statcast(start_dt='2023-03-10', end_dt='2024-03-07')
df2023 = df2023[['release_speed','batter','pitcher','description','stand','p_throws','pitch_type','balls','strikes','pfx_x','pfx_z','plate_x','plate_z','sz_top','sz_bot']]
df2023 = df2023.reset_index()


#This function calculates the distance of the pitch from the middle of the strike zone and also calculates a scaled version of this
def computeDistanceFromMiddle(df, row):
    zoneWidth = 1.41667 #width of the plate
    zoneHeight = df["sz_top"][row] - df["sz_bot"][row] #height of the zone

    scalingFactor = zoneWidth / zoneHeight #we will scale the z distance by this ratio

    zAxisMiddle = (df["sz_top"][row] + df["sz_bot"][row])/2 #The middle of strike zone along the z axis

    xDistanceFromMiddle = abs(df["plate_x"][row]) #Absolute value horizontal distance of the pitch from the middle 
    zDistanceFromMiddle = abs(df["plate_z"][row] - zAxisMiddle) #Absolute value vertical distance of the pitch from the middle
    
    zDistanceFromMiddleScaled = zDistanceFromMiddle * scalingFactor #Scale the z distance

    totalDistanceFromMiddle = math.sqrt(xDistanceFromMiddle ** 2 + zDistanceFromMiddle ** 2) #Euclidean distance formula to find total distance from middle
    totalDistanceFromMiddleScaled = math.sqrt(xDistanceFromMiddle ** 2 + zDistanceFromMiddleScaled ** 2) 
    return totalDistanceFromMiddle,totalDistanceFromMiddleScaled

#This function combines balls and strikes into one string
def combine_counts(row):
    return f"{row['balls']}-{row['strikes']}" 

#This function updates year 1 and 2 csv files to include new statistics for machine learning
def updateCSV(df, year):
    #This for loop adds in missing strike zone data
    for i in range(len(df)):
        if pd.isna(df["sz_top"][i]):
            batterDf = df[df["batter"] == df["batter"][i]]
            batterSzTop = batterDf["sz_top"].mean()
            if pd.isna(batterSzTop):
                df["sz_top"][i] = df["sz_top"].mean()
            else:
                df["sz_top"][i] = batterDf["sz_top"].mean()
        if pd.isna(df["sz_bot"][i]):
            batterDf = df[df["batter"] == df["batter"][i]]
            batterSzBot = batterDf["sz_bot"].mean()
            if pd.isna(batterSzBot):
                df["sz_bot"][i] = df["sz_bot"].mean()
            else:
                df["sz_bot"][i] = batterDf["sz_bot"].mean()
    df = df.dropna().reset_index() #drop na rows
    
    totalDistanceFromMiddleList = []
    totalDistanceFromMiddleScaledList = []
    didSwing = []
    pitchTypeClassification = []

    for i in range(len(df)):
        distanceCalculations = computeDistanceFromMiddle(df,i) #Calculate the distance of the pitch from the middle
        totalDistanceFromMiddleList.append(distanceCalculations[0]) #Add the distance to our list of distances
        totalDistanceFromMiddleScaledList.append(distanceCalculations[1])

        if df["description"][i] in ["ball","blocked_ball","called_strike","hit_by_pitch","pitchout"]: #checks if there was no swing
            didSwing.append(0) #appends 0 for no swing
        else:
            didSwing.append(1) #appends 1 for swing

        #This if-statement classifies different pitches as either a fastball, off-speed, breaking ball, or none of these. NA represents a knuckleball which is a category of its own
        if df["pitch_type"][i] in ["SI","FF","FC","FA"]:
            pitchTypeClassification.append("Fastball")
        elif df["pitch_type"][i] in ["SL","KC","CU","ST","SV","CS","SC"]:
            pitchTypeClassification.append("Breaking Ball")
        elif df["pitch_type"][i] in ["CH","EP","FO"]:
            pitchTypeClassification.append("Offspeed")
        else:
            pitchTypeClassification.append("NA")


    df["distanceFromMiddle"] = totalDistanceFromMiddleList
    df["distanceFromMiddleScaled"] = totalDistanceFromMiddleScaledList

    df["pitchTypeClassification"] = pitchTypeClassification
    df["didSwing"] = didSwing


    df['count'] = df.apply(combine_counts, axis=1) #create column for the count by combining balls and strikes

    #Create duplicate columns for the variables that we are going to make dummies from as this process removes the original column
    df['countOriginal'] = df['count']
    df['pitch_typeOriginal'] = df['pitch_type']
    df["pitchTypeClassificationOriginal"] = df["pitchTypeClassification"] 

    #Create dummies for count, pitch_type, and pitchTypeClassification variables
    df = pd.get_dummies(df, columns=['count'], prefix='count')
    df = pd.get_dummies(df, columns=['pitch_type'], prefix='pitch_type')
    df = pd.get_dummies(df, columns=['pitchTypeClassification'], prefix='pitchClassification')

    df.to_csv("df" + str(year) + ".csv", index=False) #Send this updated dataframe to a csv


warnings.filterwarnings("ignore")

updateCSV(df2022, 2022)
updateCSV(df2023, 2023)

