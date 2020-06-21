import pandas as pd
import numpy as np
data = pd.read_csv('step1_ballByballData.csv')

## Get all the players who batted in a match
#print (data["batsman"].unique())

allBatsman = data["batsman"].unique()
# For a single match

dt=pd.DataFrame()

lno_of_matches=[]
player=[]

lavg_asBataman=[]
lavg_asBowler=[]
lavg_asFielder=[]
lavg_d11sc=[]
lstd_asBataman=[]
lstd_asBowler=[]
lstd_asFielder=[]
lstd_d11sc=[]

allBatsman={"MC Henriques", "BCJ Cutting", "DJ Hooda", "CH Gayle", "KM Jadhav", "Mandeep Singh", "S Dhawan", "TM Head", "Yuvraj Singh"}

for PlayerName in allBatsman:

	try:
		df = pd.read_csv("Player_Scores//{}.csv".format(PlayerName))
	except:
		print('continue')        
		continue
    
	no_of_matches = len(df)
	avg_asBataman = df["asBataman"].mean()
	avg_asBowler = df["asBowler"].mean()
	avg_asFielder = df["asFielder"].mean()
	avg_d11sc = df["Dream11Score"].mean()

	std_asBataman = df["asBataman"].std()
	std_asBowler = df["asBowler"].std()
	std_asFielder = df["asFielder"].std()
	std_d11sc = df["Dream11Score"].std()


	lno_of_matches.append(no_of_matches)

	lavg_asBataman.append(avg_asBataman)
	lavg_asBowler.append(avg_asBowler)
	lavg_asFielder.append(avg_asFielder)
	lavg_d11sc.append(avg_d11sc)

	lstd_asBataman.append(std_asBataman)
	lstd_asBowler.append(std_asBowler)
	lstd_asFielder.append(std_asFielder)
	lstd_d11sc.append(std_d11sc)

	player.append(PlayerName)
    
	lastBatsman=PlayerName
	pd.DataFrame({"Last_batsman":PlayerName},index=range(0,1)).to_csv("Last.csv",index=False)

dt["Matches"] = lno_of_matches
dt["Player"] = player
dt["Avg_Bats"] = lavg_asBataman
dt["Avg_Bowler"] = lavg_asBowler
dt["Avg_field"] = lavg_asFielder
dt["Avg_Total"] = lavg_d11sc
dt["stdBat"] = lstd_asBataman
dt["stdBowl"] = lstd_asBowler
dt["stdFiekd"] = lstd_asFielder
dt["stdDr11"] = lstd_d11sc

dt.to_csv("PlayerAvgDreamScores.csv")
