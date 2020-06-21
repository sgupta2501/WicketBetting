import pandas as pd
import numpy as np

#read_orig_file():
data = pd.read_csv('ballByballData.csv')
#Get all the team names

'''
column_values = data[["batting_team","bowling_team"]].values.ravel()
unique_values =  pd.unique(column_values)

df = pd.DataFrame({'Team': unique_values})
df.to_csv("ListAllPairs.csv",index=False)
'''
df.drop_duplicates().to_dict("records")
