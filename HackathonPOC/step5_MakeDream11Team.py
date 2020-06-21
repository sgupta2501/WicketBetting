
'''
Rules
-------------

***RULES***

> - Total usable credit is 100, As soon as you have enough credit you can select players.

> - Only one WK is allowed
> - Batsman: Min 5 , Max 7
> - All Rounder (AR): Min 1 , Max 3
> - Bowler: Min 3, Max 5
'''

from itertools import combinations 
names=[]
name_players=[]
team=[]
wk=[]
bowl=[]
batsman=[]
ar=[]

f = open("step5_finalcreditScores.txt", "r")


for name in f.readlines():
    name_players.append( [name.lower() for name in (name.split())])
f.close()
#print(name_players) # stores info of all players taken from txt file

try:
    for player_info in name_players:
        if player_info[2]=='wk':
            wk.append(player_info)
        if player_info[2]=='bowl':
            bowl.append(player_info)
        if player_info[2]=='batsman':
            batsman.append(player_info)
        if player_info[2]=='ar':
            ar.append(player_info)
except:
    pass
        
#print(list(set(team)))
#team_1=list(set(team))[0]
#team_2=list(set(team))[1]
#print(team_1)
#print(team_2)
#print(wk)
#print(bowl)
#print(batsman)
#print(ar)


def comb(n,c):
    comb_num=[i for i in range(n)]
    comb=combinations(comb_num,c)
    return (list(comb))

bowl_list_struct=[]
batsman_list_struct=[]
ar_list_struct=[]
bowl_list=[]
batsman_list=[]
ar_list=[]
bowl_list_player=[]
batsman_list_player=[]
ar_list_player=[]



bowl_list_struct.append([comb(len(bowl),3),comb(len(bowl),4),comb(len(bowl),5)])
for x in bowl_list_struct[0]:
    for y in x:
        bowl_list.append(y)
for seq in bowl_list:
    bowl_list_player.append([bowl[index] for index in seq])
        
batsman_list_struct.append([comb(len(batsman),3),comb(len(batsman),4),comb(len(batsman),5)])
for x in batsman_list_struct[0]:
    for y in x:
        batsman_list.append(y)
for seq in batsman_list:
    batsman_list_player.append([batsman[index] for index in seq])


ar_list_struct.append([comb(len(ar),1),comb(len(ar),2),comb(len(ar),3)])
for x in ar_list_struct[0]:
    for y in x:
        ar_list.append(y)
for seq in ar_list:
    ar_list_player.append([ar[index] for index in seq])

comb_list_1=[]
for wk_player in wk:
    for batsman_player in batsman_list_player:
        for ar_player in ar_list_player:
            for bowl_player in bowl_list_player:
                comb_list_1.append([wk_player,batsman_player,ar_player,bowl_player])

comb_list_2=[]
for comb in comb_list_1:
    comb_list_2_temp=[]
    comb_list_2_temp.append(comb[0])
    for comb_batsman in comb[1]:
        comb_list_2_temp.append(comb_batsman)
    for comb_bowl in comb[3]:
        comb_list_2_temp.append(comb_bowl)
    if len(comb[2])<=3:
        for comb_ar in comb[2]:
            comb_list_2_temp.append(comb_ar)
    if len(comb[2])==4:
        comb_list_2_temp.append(comb[2])
    comb_list_2.append([temp for temp in comb_list_2_temp ])

comb_list_3=[]

count=1
for comb in comb_list_2:
    if len(comb)==11:
        score=0 # add scores of a team
        if score<=100.0:   
            comb_list_3.append(comb)

## All Teams are in comb_list_3 list, you can save it in text file or do whatever you want with it.
count=1
print(len(comb_list_3))

for i in range(5): 
    comb=comb_list_3[i]
    print("team ")
    print (count)
    count=count+1
    for x in range(len(comb)):
        print(comb[x]),         
