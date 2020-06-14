import time
#start_time = time.time()
import pandas as pd
import numpy as np

def read_orig_file():
    data = pd.read_csv('ballByballDataCleaned.csv')
    ## Get all the players, batsman, bowlers
    allBatsman = data["batsman"].unique()
    allBowler = data["bowler"].unique()
    allPlayer=np.append(allBatsman, allBowler)
    allPlayer=np.unique(allPlayer)
    df = pd.DataFrame({'Batsman': allBatsman})
    df.to_csv("ListBatsman.csv",index=False)
    df = pd.DataFrame({'Bowler': allBowler})
    df.to_csv("ListBowler.csv",index=False)
    df = pd.DataFrame({'AllPlayer': allPlayer})
    df.to_csv("ListAllPlayer.csv",index=False)

def update_label_data_file():
    data = pd.read_csv('ballByballDataCleaned.csv')
    #print("in update_data_file")
    #player_dismissed,Wicket_in_this_over
    player_dismissed = data["player_dismissed"]
    Wicket_in_this_over=pd.notna(player_dismissed)
    Wicket_in_this_over = list(map(int, Wicket_in_this_over)) 
    data["Wicket_in_this_over"] = Wicket_in_this_over
    data.to_csv("ballByballDataCleaned.csv", index=False)

def update_last_next_wicket_over():
    data = pd.read_csv('ballByballDataCleaned.csv')
    
    match_id = data["match_id"].unique()
    for k in match_id:
        for inning in range(1,2):
            over_wicket_taken=data[(data["match_id"]== k) & (data["inning"]== inning) & (data["Wicket_in_this_over"] == 1)]['over']
            over_wicket_taken=over_wicket_taken.unique()
            
            start_over=1
            start_over1=1
            Over_of_last_wicket=0
            for j in range(0, len(over_wicket_taken)):
                for i in range(start_over, over_wicket_taken[j]+1):
                    data.loc[(data["match_id"]== k) & (data["inning"]== inning) & (data["over"]== i), ('Over_of_last_wicket')] = Over_of_last_wicket
                start_over=over_wicket_taken[j]+1
                Over_of_last_wicket=over_wicket_taken[j]
            
                Over_of_next_wicket=over_wicket_taken[j]         
                for i in range(start_over1, over_wicket_taken[j]):
                    data.loc[(data["match_id"]== k) & (data["inning"]== inning) & (data["over"]== i), ('Over_of_next_wicket')] = Over_of_next_wicket
                start_over1=over_wicket_taken[j]
            
      
    data.to_csv("ballByballDataCleaned.csv", index=False)

    '''
    over_wicket_taken=data[(data["match_id"]== 1) & (data["inning"]== 1) & (data["Wicket_in_this_over"] == 1)]['over']
    over_wicket_taken=over_wicket_taken.unique()
            
    start_over=1
    start_over1=1
    Over_of_last_wicket=0

    for j in range(0, len(over_wicket_taken)):
        print("over", "over last wicket")
        for i in range(start_over, over_wicket_taken[j]+1):
            print(i, Over_of_last_wicket)
        start_over=over_wicket_taken[j]+1
        Over_of_last_wicket=over_wicket_taken[j]
            
        Over_of_next_wicket=over_wicket_taken[j]         
        print("over", "Over_of_next_wicket")
        for i in range(start_over1, over_wicket_taken[j]):
            print(i, Over_of_next_wicket)
        start_over1=over_wicket_taken[j]
    '''
    
    
def other_work():
    # Runs scored by a player
    Sr_No = 1
    #for k in range(Sr_No ,len(allBatsman)):
    for k in range(Sr_No ,10):
        PlayerName = allBatsman[k]
        pd.DataFrame({"Last_batsman":PlayerName, "Sr_No":k},index=range(0,1)).to_csv("Lastp.csv",index=False)
        print (PlayerName,k)

        ldf = pd.DataFrame()
        Match_id = []
        Name = []
        Runs = []; 	# No. of runs scored by a batsman in a match
        boPld = []  # No. of balls played by a batsman in a match
        chauke =[] 	# No. of fours a batsman hit in a match
        chhakke = []  # No. of sixes
        catches = []
        runout = []  # How many runouts a player have got
        strkrate = []  # Strike rate of a batsman in a match

        lwide_runs=[]
        lbye_runs=[]
        llegbye_runs=[]
        lnoball_runs=[]
        lpenalty_runs=[]
        lextra_runs	=[]
        ltotal_runs=[]
        lovers = []
        lwicket = []
        leco_rate = [] 
        lasBataman = []
        lasBowler = []
        lasFielder = []
        d11sc = []

        for mid in range(1,637):
            print ("Player Choosen: ", PlayerName)
            # PlayerName = input("\nEnter the name of the player ")
            print ("Wait...We are generating player's stats for match", mid)

            ## Batting score
            ballsPlayed = data[(data["batsman"]== PlayerName) &(data['match_id']==mid)]["batsman_runs"].count()
            if ballsPlayed == 0:
                continue
            RunPlayer = data[(data["batsman"]== PlayerName) &(data['match_id']==mid)]["batsman_runs"].sum()
            SR = RunPlayer/ballsPlayed
            Four_byPlayer = data[(data["batsman"]== PlayerName)& (data['match_id']==mid) & (data["batsman_runs"]== 4)]["batsman_runs"].count()
            Six_byPlayer = data[(data["batsman"]== PlayerName)& (data['match_id']==mid) & (data["batsman_runs"]== 6)]["batsman_runs"].count()

            if RunPlayer >= 100:
                bonusScore = 8
            elif RunPlayer >=50:
                bonusScore = 4
            else:
                bonusScore = 0

            if SR <= 0.5:
                penaltyScore = -3
            elif SR > 0.5 and SR <=0.599:
                penaltyScore = -2
            elif SR>=0.6 and SR <= 0.7:
                penaltyScore = -1
            else:
                penaltyScore = 0


            ## Fielding Score
            cfieldScore = data[(data["fielder"]==PlayerName) & (data['match_id']==mid) & (data["dismissal_kind"]=="caught")]['dismissal_kind'].count()
            rfieldScore = data[(data["fielder"]==PlayerName) & (data['match_id']==mid) & (data["dismissal_kind"]=="run out")]['dismissal_kind'].count()
            fieldScore = cfieldScore*4+rfieldScore*4


            ## Bowling score
            bfieldScore = data[(data["bowler"]==PlayerName) &(data['match_id']==mid) &(data["dismissal_kind"]=="bowled")]['dismissal_kind'].count()

            overs = len(data[(data["bowler"]== PlayerName) &(data['match_id']==mid)]["over"].unique())
            wicket = data[(data["bowler"]== PlayerName) &(data['match_id']==mid)]["player_dismissed"].count()

            wide_runs = data[(data["bowler"]== PlayerName) &(data['match_id']==mid)]["wide_runs"].sum()
            bye_runs =  data[(data["bowler"]== PlayerName) &(data['match_id']==mid)]["bye_runs"].sum()
            legbye_runs = data[(data["bowler"]== PlayerName) &(data['match_id']==mid)]["legbye_runs"].sum()
            noball_runs = data[(data["bowler"]== PlayerName) &(data['match_id']==mid)]["noball_runs"].sum()
            penalty_runs = data[(data["bowler"]== PlayerName) &(data['match_id']==mid)]["penalty_runs"].sum()
            extra_runs = data[(data["bowler"]== PlayerName) &(data['match_id']==mid)]["extra_runs"].sum()
            total_runs = data[(data["bowler"]== PlayerName) &(data['match_id']==mid)]["total_runs"].sum()


            if overs == 0:
                eco_rate = 0
                bpenaltyScore = 0
            else:
                eco_rate = total_runs/(overs)
                if eco_rate <= 4:
                    bpenaltyScore = 3
                elif eco_rate > 4 and eco_rate <= 5:
                    bpenaltyScore = 2
                elif eco_rate>=5 and eco_rate <= 6:
                    bpenaltyScore = 1
                elif eco_rate>=9 and eco_rate <= 10:
                    bpenaltyScore = -1
                elif eco_rate>=10 and eco_rate <= 11:
                    bpenaltyScore = -2
                elif eco_rate>=11:
                    bpenaltyScore = -3
                else:
                    bpenaltyScore = 0


            if wicket == 4:
                sc_score = 4
            elif wicket > 4:
                sc_score = 5	
            else:
                sc_score = 0

            asBataman = RunPlayer*0.5+Four_byPlayer*0.5+Six_byPlayer+bonusScore+penaltyScore
            asBowler = wicket*10 + bpenaltyScore + sc_score 
            asFielder = fieldScore
            ## Dream 11 score
            dr11Score = asBataman + asBowler + asFielder 
            print ("batsman:",PlayerName,"Runs",RunPlayer,"Balls played" ,ballsPlayed,"4s",Four_byPlayer, "6s",Six_byPlayer,"Strike Rate",SR,"\nDream11 Score", dr11Score)


        # Appending all the data to lists to finally give it to panda and write on csv/excel file
            Match_id.append(mid)
            Name.append(PlayerName)
            Runs.append(RunPlayer)
            boPld.append(ballsPlayed)
            chauke.append(Four_byPlayer) 
            chhakke.append(Six_byPlayer)
            strkrate.append((int(SR*100)))
            catches.append(cfieldScore) 
            runout.append(rfieldScore)

            lovers.append(overs)
            lwicket.append(wicket)
            leco_rate.append(eco_rate)
            lwide_runs.append(wide_runs)
            lbye_runs.append(bye_runs)
            llegbye_runs.append(legbye_runs)
            lnoball_runs.append(noball_runs)
            lpenalty_runs.append(penalty_runs)
            lextra_runs	.append(extra_runs)
            ltotal_runs.append(total_runs)

            lasBataman.append(asBataman)
            lasBowler.append(asBowler)
            lasFielder.append(asFielder)
            d11sc.append(dr11Score)



        ldf["Sr_No"] = [i for i in range(len(Name))]
        ldf["MatchID"] = Match_id
        ldf["Name"] = Name
        ldf["Runs"] = Runs
        ldf["Balls"] = boPld
        ldf["4s"] = chauke
        ldf["6s"] = chhakke
        ldf["Strike_Rate"] = strkrate
        ldf["Catches"] = catches
        ldf["Run_Out"] = runout

        ldf["over"] = lovers
        ldf["total_runs"] = ltotal_runs
        ldf["wicket"] = lwicket
        ldf["economy"] = leco_rate
        ldf["wide_runs"] = lwide_runs
        ldf["bye_runs"] = lbye_runs
        ldf["legbye_runs"] = llegbye_runs
        ldf["noball_runs"] = lnoball_runs
        ldf["penalty_runs"] = lpenalty_runs
        ldf["extra_runs"] = lextra_runs

        ldf["asBataman"] = lasBataman
        ldf["asBowler"] = lasBowler
        ldf["asFielder"] = lasFielder

        ldf["Dream11Score"] = d11sc


        ldf.to_csv("Player_Scores\\{}.csv".format(PlayerName),index=False)
    '''
        print ("===========================================================================")
        print ("Time taken in the process", time.time()-start_time)
        print ("===========================================================================")

    '''



