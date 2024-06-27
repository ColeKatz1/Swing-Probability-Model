import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable


#SWING PROBABILITY BY PITCH LOCATION
df = pd.read_csv("df2023WithPredictions.csv")
df = df.dropna(subset=["plate_x","plate_z"])

plt.figure()
hist, xEdges, yEdges = np.histogram2d(df['plate_x'], df['plate_z'], bins=70, weights=df['SwingProbability'])

x_centers = (xEdges[:-1] + xEdges[1:]) / 2
y_centers = (yEdges[:-1] + yEdges[1:]) / 2
X, Y = np.meshgrid(x_centers, y_centers)

plt.pcolormesh(X, Y, hist.T, cmap='YlGnBu')

#Create the colorbar
sm = ScalarMappable(cmap='YlGnBu', norm=Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Swing Probability')


avgTop = df["sz_top"].mean()
avgBot = df["sz_bot"].mean()

plt.xlabel('Plate X')
plt.ylabel('Plate Z')
plt.title('Swing Probability By Pitch Location')

#Create the strike zone
plt.plot([-.7, -.7], [avgBot, avgTop], color='black', linestyle='--')  
plt.plot([.7, .7], [avgBot, avgTop], color='black', linestyle='--')  
plt.plot([-.7, .7], [avgTop, avgTop], color='black', linestyle='--')  
plt.plot([-.7, .7], [avgBot, avgBot], color='black', linestyle='--')  

plt.ylim(0,4.2)
plt.xlim(-2.5,2.5)


#SWING PERCENTAGE BY COUNT
df2022 = pd.read_csv("df2022.csv")
df2023 = pd.read_csv("df2023.csv")

combinedDf = pd.concat([df2022, df2023], ignore_index=True) #combined year 1 and 2 into 1 df
groupedDf = combinedDf.groupby(["countOriginal"]).agg({'didSwing': 'mean'}).reset_index().sort_values(by = "didSwing", ascending=False) #Find swing percentage grouped by count
plt.figure()
plt.grid(zorder=0, axis='y', linestyle='-', linewidth=0.5, color = 'black') #Add the gridlines
plt.bar(groupedDf["countOriginal"][1:-1],groupedDf["didSwing"][1:-1]*100, color = 'dodgerblue', edgecolor = 'black', zorder=3) #Create bar plot
plt.title('Swing Percentage by Count', fontweight='bold') #Add title
plt.xlabel('Count', fontweight='bold') #Add x-axis label
plt.ylabel('Swing Percentage', fontweight='bold') #Add y-axis label
#print(groupedDf)

#SWING PERCENTAGE BY PITCH TYPE

groupedDf = combinedDf.groupby(["pitch_typeOriginal"]).agg({'didSwing': 'mean','batter': 'count'}).reset_index().sort_values(by = "didSwing", ascending=False) #Find swing percentage grouped by pitch type
plt.figure()
plt.grid(zorder=0, axis='y', linestyle='-', linewidth=0.5, color = 'black')
plt.bar(groupedDf["pitch_typeOriginal"][1:],groupedDf["didSwing"][1:]*100, color = 'dodgerblue', edgecolor = 'black', zorder=3) 
plt.title('Swing Percentage by Pitch Type', fontweight='bold') 
plt.xlabel('Pitch Type', fontweight='bold')
plt.ylabel('Swing Percentage', fontweight='bold')

#SWING PERCENTAGE BY PITCH CLASSIFICATION

groupedDf = combinedDf.groupby(["pitchTypeClassificationOriginal"]).agg({'didSwing': 'mean','batter': 'count'}).reset_index().sort_values(by = "didSwing", ascending=False) #Find swing percentage grouped by pitch classificaiton
plt.figure()
plt.grid(zorder=0, axis='y', linestyle='-', linewidth=0.5, color = 'black')
plt.bar(groupedDf["pitchTypeClassificationOriginal"],groupedDf["didSwing"]*100, color = 'dodgerblue', edgecolor = 'black', zorder=3) 
plt.title('Swing Percentage by Pitch Classification', fontweight='bold') 
plt.xlabel('Pitch Classification', fontweight='bold')
plt.ylabel('Swing Percentage', fontweight='bold')

#OVERALL SWING PERCENTAGE

groupedDf = combinedDf.groupby(["didSwing"]).agg({'batter': 'count'}).reset_index()
#print(groupedDf)
plt.figure()
plt.grid(zorder=0, axis='y', linestyle='-', linewidth=0.5, color = 'black')
plt.bar(["No Swing","Swing"],groupedDf["batter"]/groupedDf["batter"].sum()*100, color = 'dodgerblue', edgecolor = 'black', zorder=3) 
plt.title('Overall Pitch Swing Percentage', fontweight='bold') 
plt.ylabel('Percentage', fontweight='bold')

#MIDDLE MIDDLE SWING PERCENTAGE

middleMiddleDf = combinedDf[(combinedDf['distanceFromMiddle'] < .5)]
groupedDf = middleMiddleDf.groupby(["didSwing"]).agg({'batter': 'count'}).reset_index()
#print(groupedDf)
plt.figure()
plt.grid(zorder=0, axis='y', linestyle='-', linewidth=0.5, color = 'black')
plt.bar(["No Swing","Swing"],groupedDf["batter"]/groupedDf["batter"].sum()*100, color = 'dodgerblue', edgecolor = 'black', zorder=3) 
plt.title('Middle-Middle Pitch Swing Percentage', fontweight='bold') 
plt.ylabel('Percentage', fontweight='bold')
plt.show()
