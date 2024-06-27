FILES:

cleanData.py: This file collects pitch-by-pitch Statcast data from the 2022 and 2023 seasons and updates this data by creating new variables which 
will be important for the machine learning model like the distance of each pitch from the middle of the strike zone, count dummy variables, pitch type (changup, sluve, etc.) 
dummy variables, and pitch classification (fastball, off-speed, breaking ball) dummy variables. Additionally, missing information about a batter's strike zone height in some 
rows is filled in using other appearances of that batter.  

backtestModels.py: This file trains multiple logistic regression machine learning models on the 2022 and 2023 seasons and tests the accuracy of these models by utilizing different variations of variables.

machineLearning.py: This file trains the final logistic regression model on the 2022 season and creates predictions for the 2023 season which are then saved in a df2023WithPredictions.csv file. 

visualizations.py: This file visualizes how various factors affect swing probability. 
