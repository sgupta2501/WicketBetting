import numpy as np
import pandas as pd
import csv  
    
def wicket_betting_final_data():

    data = pd.read_csv('WicketBetting_TrainingData.csv')   
    
    Over_of_last_wicket_idx=data.columns.get_loc("Over_of_last_wicket")
    Over_of_next_wicket_idx=data.columns.get_loc("Over_of_next_wicket")
    tot_wicket_upto_this_over_idx=data.columns.get_loc("tot_wicket_upto_this_over")
    diff_last_next_wicket = data.iloc[:,Over_of_next_wicket_idx].values - data.iloc[:,Over_of_last_wicket_idx].values
    tot_wicket_till_now = data.iloc[:,tot_wicket_upto_this_over_idx].values
    p1_idx=data.columns.get_loc("p1")

    p21_idx=data.columns.get_loc("p21")
    p22_idx=data.columns.get_loc("p22")
    p23_idx=data.columns.get_loc("p23")
    p24_idx=data.columns.get_loc("p24")
    p31_idx=data.columns.get_loc("p31")
    p32_idx=data.columns.get_loc("p32")
    p33_idx=data.columns.get_loc("p33")
    p34_idx=data.columns.get_loc("p34")
    p41_idx=data.columns.get_loc("p41")
    p42_idx=data.columns.get_loc("p42")
    p43_idx=data.columns.get_loc("p43")
    p44_idx=data.columns.get_loc("p44")
    #print(p21_idx,p22_idx,p23_idx,p24_idx)
    #print(p31_idx,p32_idx,p33_idx,p34_idx)
    #print(p41_idx,p42_idx,p43_idx,p44_idx)
    y1_idx=data.columns.get_loc("wicket_in_over_1")
    y2_idx=data.columns.get_loc("wicket_in_over_2")
    y3_idx=data.columns.get_loc("wicket_in_over_3")
    y4_idx=data.columns.get_loc("wicket_in_over_4")
    
    for i in range(0, len(diff_last_next_wicket)):
        if diff_last_next_wicket[i]<=0:
            diff_last_next_wicket[i]= 1
        if tot_wicket_till_now[i]<=0:
            tot_wicket_till_now[i]= 1

    p5 = [i / j for i, j in zip(tot_wicket_till_now, diff_last_next_wicket)]
    x_p1= np.multiply(data.iloc[:,p1_idx].values, p5)

    x_1= np.insert(data.iloc[:,[p21_idx,p31_idx,p41_idx]].values, 0, x_p1, axis=1)
    x_2= np.insert(data.iloc[:,[p22_idx,p32_idx,p42_idx]].values, 0, x_p1, axis=1)
    x_3= np.insert(data.iloc[:,[p23_idx,p33_idx,p43_idx]].values, 0, x_p1, axis=1)
    x_4= np.insert(data.iloc[:,[p24_idx,p34_idx,p44_idx]].values, 0, x_p1, axis=1)
    
    y= data.iloc[:,20].values

    yx_1= np.insert(x_1, 0, data.iloc[:,y1_idx].values, axis=1)
    yx_2= np.insert(x_2, 0, data.iloc[:,y2_idx].values, axis=1)
    yx_3= np.insert(x_3, 0, data.iloc[:,y3_idx].values, axis=1)
    yx_4= np.insert(x_4, 0, data.iloc[:,y4_idx].values, axis=1)

    for j in range(0,5):
        print(j, yx_1[:,j].shape)
        index = 0
        for i in yx_1[:,j]:
            if not np.isfinite(i):
                #print(index, i)
                yx_1[index,j]=0
            index +=1
    
    filename = "yx_1.csv"
    fields = ['y','p1','p21','p31','p41']
    # writing to csv file  
    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        # writing the fields  
        csvwriter.writerow(fields)  
        csvwriter.writerows(yx_1)  

    for j in range(0,5):
        print(j, yx_2[:,j].shape)
        index = 0
        for i in yx_2[:,j]:
            if not np.isfinite(i):
                #print(index, i)
                yx_2[index,j]=0
            index +=1
    
    filename = "yx_2.csv"
    fields = ['y','p1','p22','p32','p42']
    # writing to csv file  
    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        # writing the fields  
        csvwriter.writerow(fields)  
        csvwriter.writerows(yx_2)  

    for j in range(0,5):
        print(j, yx_3[:,j].shape)
        index = 0
        for i in yx_3[:,j]:
            if not np.isfinite(i):
                #print(index, i)
                yx_3[index,j]=0
            index +=1
    
    filename = "yx_3.csv"
    fields = ['y','p1','p23','p33','p43']
    # writing to csv file  
    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        # writing the fields  
        csvwriter.writerow(fields)  
        csvwriter.writerows(yx_3)  

    for j in range(0,5):
        print(j, yx_4[:,j].shape)
        index = 0
        for i in yx_4[:,j]:
            if not np.isfinite(i):
                #print(index, i)
                yx_4[index,j]=0
            index +=1
    
    filename = "yx_4.csv"
    fields = ['y','p1','p24','p34','p44']
    # writing to csv file  
    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        # writing the fields  
        csvwriter.writerow(fields)  
        csvwriter.writerows(yx_4)  
        
    '''
    print('If diff is negative ???')
    d=data.iloc[:,Over_of_last_wicket_idx]
    print("totl entries ", len(d))
    print("col no for Over_of_last_wicket", Over_of_last_wicket_idx)
    
    neg=0
    zer=0
    for i in range(0, len(d)):
        if data.iloc[i,Over_of_last_wicket_idx]<0:
            neg += 1
        elif data.iloc[i,Over_of_last_wicket_idx]==0:
            zer += 1
    print("neg ",neg, "zero ", zer)


    d=data.iloc[:,Over_of_next_wicket_idx]
    print("col no for Over_of_next_wicket", Over_of_next_wicket_idx)
    
    neg=0
    zer=0
    for i in range(0, len(d)):
        if data.iloc[i,Over_of_next_wicket_idx]<0:
            neg += 1
        elif data.iloc[i,Over_of_next_wicket_idx]==0:
            zer += 1
    print("neg ",neg, "zero ", zer)
    
    print("totl entries ", len(diff_last_next_wicket))
    
    neg=0
    zer=0
    for i in range(0, len(diff_last_next_wicket)):
        if diff_last_next_wicket[i]<0:
            neg += 1
        elif diff_last_next_wicket[i]==0:
            zer += 1
    print("neg ",neg, "zero ", zer)

    neg=0
    zer=0
    for i in range(0, len(tot_wicket_till_now)):
        if tot_wicket_till_now[i]<0:
            neg += 1
        elif tot_wicket_till_now[i]==0:
            zer += 1
    print("neg ",neg, "zero ", zer)
    '''
    
    
    
        
def batsman_prob_in_a_over(batsman, over):
    if over > 20:
        num=0
        den=1
    else:
        data_batsman = pd.read_csv('Batsman_OutPerOver.csv')
        str_out="out_over_"+ str(over)

        num=data_batsman[(data_batsman["batsman"]== batsman)][str_out].values
        den=data_batsman[(data_batsman["batsman"]== batsman)]["Tot_Match_Played"].values

    if (den==0):
        p=0
    else:
        p=num/den
    #print(p)
    return float(p)


def bowler_prob_in_a_over(bowler, over):
    
    if over > 20:
        num=0
        den=1
    else:
        data_bowler = pd.read_csv('Bowler_WicketsPerOver.csv')
        str_wicket="Wicket_Over_"+ str(over)

        #d=list(data_bowler.iloc[:,0])
        #bowler=data_bowler.iloc[2,0]
        #print(d)
        #for i in range (0, len(d)):
            #if (d[i]==bowler):
                #print("is present")

        den=data_bowler[(data_bowler["bowler"]== bowler)]["Tot_Match_Played"].values
        num=data_bowler[(data_bowler["bowler"]== bowler)][str_wicket].values

    if (den==0):
        p1=0
    else:
        p1=num/den
    #print(p1)
    return float(p1)
    
    
def make_training_data():
    data = pd.read_csv('ballByballDataCleaned.csv')
    #matchNo=1:636
    #over=1:12
    #inning=1:2

    df=pd.DataFrame()
    tot_wicket_upto_this_over_old=0
    for rowNo in range (0,150456):
    #for rowNo in range (73813,150456):
    #for rowNo in range (0,110):
        d1=list(data.iloc[rowNo,:])
        print(rowNo)
        if (d1[3] != 1):
            continue
        rowNo +=6    
        bowler=d1[6]
        batsman=d1[4]
        non_striker=d1[5]
        over=d1[2]
        Over_of_last_wicket=d1[10]
        Over_of_next_wicket=d1[11]
        tot_wicket_upto_this_over=d1[12]
        wicket_in_this_over=(tot_wicket_upto_this_over-tot_wicket_upto_this_over_old)%2
        tot_wicket_upto_this_over_old=tot_wicket_upto_this_over

        #print(rowNo, over, bowler)

        # find prob (p1) of wicket by this bowler in this over

        if (over >20):
            continue

        p1=bowler_prob_in_a_over(bowler, over)

        p21=bowler_prob_in_a_over(bowler, over+1)
        p22=bowler_prob_in_a_over(bowler, over+2)
        p23=bowler_prob_in_a_over(bowler, over+3)
        p24=bowler_prob_in_a_over(bowler, over+4)

        p31=batsman_prob_in_a_over(batsman, over+1)
        p32=batsman_prob_in_a_over(batsman, over+2)
        p33=batsman_prob_in_a_over(batsman, over+3)
        p34=batsman_prob_in_a_over(batsman, over+4)

        p41=batsman_prob_in_a_over(non_striker, over+1)
        p42=batsman_prob_in_a_over(non_striker, over+2)
        p43=batsman_prob_in_a_over(non_striker, over+3)
        p44=batsman_prob_in_a_over(non_striker, over+4)
        
        #print(bowler, batsman, non_striker, over, Over_of_last_wicket, Over_of_next_wicket, tot_wicket_upto_this_over, p1, p21, p31, p41, p22, p32, p42, p23, p33, p43, p24, p34, p44)
        new_row={"bowler":bowler,"batsman":batsman,"non_striker":non_striker, "over":over, "Over_of_last_wicket":Over_of_last_wicket, "Over_of_next_wicket":Over_of_next_wicket, "tot_wicket_upto_this_over":tot_wicket_upto_this_over, "p1":p1, "p21":p21,"p31":p31,"p41":p41, "p22":p22,"p32":p32,"p42":p42,"p23":p23,"p33":p33,"p43":p43,"p24":p24,"p34":p34,"p44":p44,"wicket_in_this_over":wicket_in_this_over}
        df = df.append(new_row, ignore_index=True)                            
        
        df.to_csv("WicketBetting_TrainingData.csv",index=False)    
    
    #df.to_csv("WicketBetting_TrainingData.csv",index=False)    

    
def add_wicket_in_next4_over():
    
    data = pd.read_csv('ballByballDataCleaned.csv')
    data1 = pd.read_csv('WicketBetting_TrainingData.csv')
    #match_id,inning,over,ball,batsman,non_striker,bowler,total_runs,player_dismissed,Wicket_in_this_over,Over_of_last_wicket,Over_of_next_wicket,tot_wicket_upto_this_over,wicket_in_over_1,wicket_in_over_2,wicket_in_over_3,wicket_in_over_4

    #TM Head,BCJ Cutting,KM Jadhav,9
    #1,2,9,1,TM Head,KM Jadhav,BCJ Cutting,1,,0,7,12,2.0,0.0,0.0,1.0,1.0
    print(len(data1))
    for i in range(0, len(data1)):
    #for i in range(0, 1):
        print(i)
        d=data1.iloc[i,:]
        batsman=d["batsman"]
        bowler=d["bowler"]
        non_striker=d["non_striker"]

        wicket_in_over_1=data[(data["batsman"]== batsman) & (data["bowler"]== bowler) & (data["non_striker"] == non_striker) & (data["ball"] == 1)]['wicket_in_over_1'].values
        wicket_in_over_2=data[(data["batsman"]== batsman) & (data["bowler"]== bowler) & (data["non_striker"] == non_striker) & (data["ball"] == 1)]['wicket_in_over_2'].values
        wicket_in_over_3=data[(data["batsman"]== batsman) & (data["bowler"]== bowler) & (data["non_striker"] == non_striker) & (data["ball"] == 1)]['wicket_in_over_3'].values
        wicket_in_over_4=data[(data["batsman"]== batsman) & (data["bowler"]== bowler) & (data["non_striker"] == non_striker) & (data["ball"] == 1)]['wicket_in_over_4'].values

        data1.loc[(data1["batsman"]== batsman) & (data1["bowler"]== bowler) & (data1["non_striker"] == non_striker), ('wicket_in_over_1')] = wicket_in_over_1
        data1.loc[(data1["batsman"]== batsman) & (data1["bowler"]== bowler) & (data1["non_striker"] == non_striker), ('wicket_in_over_2')] = wicket_in_over_2
        data1.loc[(data1["batsman"]== batsman) & (data1["bowler"]== bowler) & (data1["non_striker"] == non_striker), ('wicket_in_over_3')] = wicket_in_over_3
        data1.loc[(data1["batsman"]== batsman) & (data1["bowler"]== bowler) & (data1["non_striker"] == non_striker), ('wicket_in_over_4')] = wicket_in_over_4

    data1.to_csv("WicketBetting_TrainingData.csv",index=False)    
    
def removenan():
    data = pd.read_csv('WicketBetting_TrainingData.csv')
    data["wicket_in_over_1"] = data["wicket_in_over_1"].fillna(0)
    data["wicket_in_over_2"] = data["wicket_in_over_2"].fillna(0)
    data["wicket_in_over_3"] = data["wicket_in_over_3"].fillna(0)
    data["wicket_in_over_4"] = data["wicket_in_over_4"].fillna(0)
    data.to_csv("WicketBetting_TrainingData.csv",index=False)    
    
def main():
    print("in main")
    #make_training_data()
    #add_wicket_in_next4_over()
    #removenan()
    #wicket_betting_final_data()
    
    