import pandas as pd
import numpy as np
import csv  
    
data = pd.read_csv('ballByballDataCleaned.csv')

def order_running():
    
    #read_orig_file()
    #update_label_data_file()
    update_last_next_wicket_over()
    update_tot_wickets_upto_over()
    #batsman_over_got_out()
    #batsman_runs_obtained()
    #bowler_wickets_taken()
    
def update_tot_wickets_upto_over():
    
    #data = pd.read_csv('ballByballDataCleaned.csv')
    match_id = data["match_id"].unique()
    for k in match_id:
    #for k in range (1,2):
        for inning in range(1,3):
            tot_wicket_upto_this_over=0
            for over in range(1,21):
                Wicket_in_this_over=data[(data["match_id"]== k) & (data["inning"]== inning) & (data["over"] == over)]['Wicket_in_this_over']
                tot_wicket_upto_this_over += sum(Wicket_in_this_over)
                data.loc[(data["match_id"]== k) & (data["inning"]== inning) & (data["over"]== over), ('tot_wicket_upto_this_over')] = tot_wicket_upto_this_over                    

    data.to_csv("ballByballDataCleaned.csv", index=False)

def batsman_over_got_out():

    #data = pd.read_csv('ballByballDataCleaned.csv')
    batsman = data["batsman"].unique()
    match_id = data["match_id"].unique()
    df=pd.DataFrame()
    for k in range(0, len(batsman)):
    #for k in range(0, 1):
        print(batsman[k])
        matches=data[(data["batsman"]== batsman[k])]["match_id"].unique()
        NoOfMatches= len(matches)
        
        over_where_out=data[(data["batsman"]== batsman[k]) & (data["Wicket_in_this_over"] == 1)]['over'].unique()
        match_where_out=data[(data["batsman"]== batsman[k]) & (data["Wicket_in_this_over"] == 1)]['match_id'].unique()
        No_of_NotOut_Match=NoOfMatches - len(match_where_out)
           
        no_of_out_per_over=[0]*20
        for j in over_where_out:         
            tmp = data[(data["batsman"]== batsman[k]) & (data["Wicket_in_this_over"] == 1) & (data['over'] == j)]['match_id']
            #print(j)
            no_of_out_per_over[j-1]=len(tmp)

        #print(no_of_out_per_over)
        new_row={"batsman":batsman[k],"Tot_Match_Played":NoOfMatches,"Tot_Match_NotOut":No_of_NotOut_Match,"out_over_1":no_of_out_per_over[0],"out_over_2":no_of_out_per_over[1],"out_over_3":no_of_out_per_over[2],"out_over_4":no_of_out_per_over[3],"out_over_5":no_of_out_per_over[4],"out_over_6":no_of_out_per_over[5],"out_over_7":no_of_out_per_over[6],"out_over_8":no_of_out_per_over[7],"out_over_9":no_of_out_per_over[8],"out_over_10":no_of_out_per_over[9],"out_over_11":no_of_out_per_over[10],"out_over_12":no_of_out_per_over[11],"out_over_13":no_of_out_per_over[12],"out_over_14":no_of_out_per_over[13],"out_over_15":no_of_out_per_over[14],"out_over_16":no_of_out_per_over[15],"out_over_17":no_of_out_per_over[16],"out_over_18":no_of_out_per_over[17],"out_over_19":no_of_out_per_over[18],"out_over_20":no_of_out_per_over[19]}
        df = df.append(new_row, ignore_index=True)
        
    df.to_csv("Batsman_OutPerOver.csv",index=False)

def batsman_runs_obtained():
    
    
    #data = pd.read_csv('ballByballDataCleaned.csv')
    batsman = data["batsman"].unique()
    match_id = data["match_id"].unique()
    df=pd.DataFrame()
    for k in range(0, len(batsman)):
    #for k in range(0, 1):
        print(batsman[k])
        matches=data[(data["batsman"]== batsman[k])]["match_id"].unique()
        NoOfMatches= len(matches)
                 
        run_per_over=[0]*20
        for j in range(0, 20):
            tmp = data[(data["batsman"]== batsman[k]) & (data['over'] == j)]['total_runs']
            #print(tmp)
            run_per_over[j-1]=sum(tmp)

        tot_runs = sum(run_per_over)
        #print(run_per_over)
        #print(tot_runs)
        #print(NoOfMatches)
        new_row={"batsman":batsman[k],"Tot_Match_Played":NoOfMatches,"Tot_Runs":tot_runs,"run_Over_1":run_per_over[0],"run_Over_2":run_per_over[1],"run_Over_3":run_per_over[2],"run_Over_4":run_per_over[3],"run_Over_5":run_per_over[4],"run_Over_6":run_per_over[5],"run_Over_7":run_per_over[6],"run_Over_8":run_per_over[7],"run_Over_9":run_per_over[8],"run_Over_10":run_per_over[9],"run_Over_11":run_per_over[10],"run_Over_12":run_per_over[11],"run_Over_13":run_per_over[12],"run_Over_14":run_per_over[13],"run_Over_15":run_per_over[14],"run_Over_16":run_per_over[15],"run_Over_17":run_per_over[16],"run_Over_18":run_per_over[17],"run_Over_19":run_per_over[18],"run_Over_20":run_per_over[19]}

        df = df.append(new_row, ignore_index=True)
        
    df.to_csv("Batsman_RunPerOver.csv",index=False)

def bowler_wickets_taken():
    
    #data = pd.read_csv('ballByballDataCleaned.csv')
    # field names  
    fields = ["bowler","Tot_Match_Played","Tot_Wickets","Tot_Dud_Match_Played","Wicket_Over_1","Wicket_Over_2","Wicket_Over_3","Wicket_Over_4","Wicket_Over_5","Wicket_Over_6","Wicket_Over_7","Wicket_Over_8","Wicket_Over_9","Wicket_Over_10","Wicket_Over_11","Wicket_Over_12","Wicket_Over_13","Wicket_Over_14","Wicket_Over_15","Wicket_Over_16","Wicket_Over_17","Wicket_Over_18","Wicket_Over_19","Wicket_Over_20"]

    # name of csv file  
    filename = "Bowler_WicketsPerOver.csv"
    bowler = data["bowler"].unique()
    match_id = data["match_id"].unique()

    # writing to csv file  
    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  

        # writing the fields  
        csvwriter.writerow(fields)  
   
        for k in range(0, len(bowler)):
        #for k in range(0, 2):
            #print(bowler[k])
            matches=data[(data["bowler"]== bowler[k])]["match_id"].unique()
            NoOfMatches= len(matches)

            over_with_wicket=data[(data["bowler"]== bowler[k]) & (data["Wicket_in_this_over"] == 1)]['over'].unique()
            match_with_wicket=data[(data["bowler"]== bowler[k]) & (data["Wicket_in_this_over"] == 1)]['match_id'].unique()
            No_of_Dud_Match=NoOfMatches - len(match_with_wicket)

            wicket_per_over=[0]*20
            for j in over_with_wicket:         
                tmp = data[(data["bowler"]== bowler[k]) & (data["Wicket_in_this_over"] == 1) & (data['over'] == j)]['match_id']
                #print(j)
                wicket_per_over[j-1]=len(tmp)

            tot_wickets = sum(wicket_per_over)
            row = [bowler[k],NoOfMatches,tot_wickets,No_of_Dud_Match,wicket_per_over[0],wicket_per_over[1],wicket_per_over[2],wicket_per_over[3],wicket_per_over[4],wicket_per_over[5],wicket_per_over[6],wicket_per_over[7],wicket_per_over[8],wicket_per_over[9],wicket_per_over[10],wicket_per_over[11],wicket_per_over[12],wicket_per_over[13],wicket_per_over[14],wicket_per_over[15],wicket_per_over[16],wicket_per_over[17],wicket_per_over[18],wicket_per_over[19]]

            # writing the data rows  
            csvwriter.writerow(row)     
            
    '''
    bowler = data["bowler"].unique()
    match_id = data["match_id"].unique()
    
    df=pd.DataFrame()
    for k in range(0, len(bowler)):
    #for k in range(0, 2):
        print(bowler[k])
        matches=data[(data["bowler"]== bowler[k])]["match_id"].unique()
        NoOfMatches= len(matches)
        
        over_with_wicket=data[(data["bowler"]== bowler[k]) & (data["Wicket_in_this_over"] == 1)]['over'].unique()
        match_with_wicket=data[(data["bowler"]== bowler[k]) & (data["Wicket_in_this_over"] == 1)]['match_id'].unique()
        No_of_Dud_Match=NoOfMatches - len(match_with_wicket)
           
        wicket_per_over=[0]*20
        for j in over_with_wicket:         
            tmp = data[(data["bowler"]== bowler[k]) & (data["Wicket_in_this_over"] == 1) & (data['over'] == j)]['match_id']
            #print(j)
            wicket_per_over[j-1]=len(tmp)

        tot_wickets = sum(wicket_per_over)
        new_row={"bowler":bowler[k],"Tot_Match_Played":NoOfMatches,"Tot_Wickets":tot_wickets,"Tot_Dud_Match_Played":No_of_Dud_Match,"Wicket_Over_1":wicket_per_over[0],"Wicket_Over_2":wicket_per_over[1],"Wicket_Over_3":wicket_per_over[2],"Wicket_Over_4":wicket_per_over[3],"Wicket_Over_5":wicket_per_over[4],"Wicket_Over_6":wicket_per_over[5],"Wicket_Over_7":wicket_per_over[6],"Wicket_Over_8":wicket_per_over[7],"Wicket_Over_9":wicket_per_over[8],"Wicket_Over_10":wicket_per_over[9],"Wicket_Over_11":wicket_per_over[10],"Wicket_Over_12":wicket_per_over[11],"Wicket_Over_13":wicket_per_over[12],"Wicket_Over_14":wicket_per_over[13],"Wicket_Over_15":wicket_per_over[14],"Wicket_Over_16":wicket_per_over[15],"Wicket_Over_17":wicket_per_over[16],"Wicket_Over_18":wicket_per_over[17],"Wicket_Over_19":wicket_per_over[18],"Wicket_Over_20":wicket_per_over[19]}

        df = df.append(new_row, ignore_index=True)
        
    df.to_csv("Bowler_WicketsPerOver.csv",index=False)
                     
    '''
    
def read_orig_file():
    #data = pd.read_csv('ballByballDataCleaned.csv')
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
    #data = pd.read_csv('ballByballDataCleaned.csv')
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
        for inning in range(1,3):
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

def update_next_4_over_wicket_taken():
    data = pd.read_csv('ballByballDataCleaned.csv')
    
    match_id = data["match_id"].unique()
    for k in match_id:
        for inning in range(1,3):
            over_wicket_taken=data[(data["match_id"]== k) & (data["inning"]== inning) & (data["Wicket_in_this_over"] == 1)]['over']
            over_wicket_taken=over_wicket_taken.unique()
            
            for j in range(0, 13):
                if (j+1) in over_wicket_taken:
                    wicket_in_over_1=1
                else:
                    wicket_in_over_1=0
                data.loc[(data["match_id"]== k) & (data["inning"]== inning) & (data["over"]== j), ('wicket_in_over_1')] = wicket_in_over_1
                
                if (j+2) in over_wicket_taken:
                    wicket_in_over_2=1
                else:
                    wicket_in_over_2=0  
                data.loc[(data["match_id"]== k) & (data["inning"]== inning) & (data["over"]== j), ('wicket_in_over_2')] = wicket_in_over_2

                if (j+3) in over_wicket_taken:
                    wicket_in_over_3=1
                else:
                    wicket_in_over_3=0
                data.loc[(data["match_id"]== k) & (data["inning"]== inning) & (data["over"]== j), ('wicket_in_over_3')] = wicket_in_over_3      

                if (j+4) in over_wicket_taken:
                    wicket_in_over_4=1
                else:
                    wicket_in_over_4=0
                data.loc[(data["match_id"]== k) & (data["inning"]== inning) & (data["over"]== j), ('wicket_in_over_4')] = wicket_in_over_4                
                
    data.to_csv("ballByballDataCleaned.csv", index=False)

