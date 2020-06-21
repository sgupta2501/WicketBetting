import pandas as pd
import numpy as np

#read_orig_file():
data = pd.read_csv('ballByballData.csv', low_memory=False)
#Get all the team names
allTeam = data["batting_team"].unique()
df = pd.DataFrame({'Team': allTeam})
df.to_csv("ListAllTeams.csv",index=False)
