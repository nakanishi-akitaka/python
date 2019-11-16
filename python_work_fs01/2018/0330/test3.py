#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

def print_list(listx):
    str=''
    for i in range(len(listx)):
        str+=listx[i]
    print(str)

def switch_list(i,j,listx):
    listx[i],listx[j] = listx[j],listx[i]

# choice = gouhouhu ? (player 1)
def check_player(ip,i,state):
    istep=0
    if(ip==1):
        if(state[i]!='1' and state[i]!='2' and state[i]!='3'):
            istep=0
            return istep
        for k in range(i+1,len(state)):
             if(state[k]=='1' or state[k]=='2' or state[k]=='3'):
                 istep=0
                 break
             elif(state[k]=='-'):
                 istep=k-i
                 break
        return istep
    elif(ip==2):
        if(state[i]!='a' and state[i]!='b' and state[i]!='c'):
            istep=0
            return istep
        for k in range(i-1,0,-1):
             if(state[k]=='a' or state[k]=='b' or state[k]=='c'):
                 istep=0
                 break
             elif(state[k]=='-'):
                 istep=k-i
                 break
        return istep

# search gouhoushu
def search_gouhou(ip):
    gouhou=[]
    for i in range(1,len(state)):
        istep=check_player(ip,i,state) # choice = gouhouhu ?
        if(istep!=0):
            gouhou.append(i)
    return gouhou

game=0
turn=1
state=[' ','1','2','3','-','-','-','a','b','c']
# state=[' ','1','2','a','-','-','3','b','c','-']

####print_list(state)            # print
####i=2                          # choice
####istep=check_player1(i,state) # choice = gouhouhu ? (player 1)
####switch_list(i,i+istep,state) # update
####
####print_list(state)            # print
####i=7                          # choice
####istep=check_player2(i,state) # choice = gouhouhu ? (player 2)
####switch_list(i,i+istep,state) # update
####
####print_list(state)            # print
####i=2                          # choice
####istep=check_player1(i,state) # choice = gouhouhu ? (player 1)
####switch_list(i,i+istep,state) # update
####
####print_list(state)            # print
####i=9                          # choice
####istep=check_player2(i,state) # choice = gouhouhu ? (player 2)
####switch_list(i,i+istep,state) # update
####
####gouhou1=search_gouhou(1)
####gouhou2=search_gouhou(2)
####print(gouhou1)
####print(gouhou2)
####
####for test
####if(len(gouhou2)>0):
####    for i in range(10):
####        j=random.randrange(0,len(gouhou2))
####        print(j,gouhou2[j])
for iturn in range(20):
    print_list(state)
    gouhou=search_gouhou(1)
    if(len(gouhou)>0):
        i=gouhou[random.randrange(0,len(gouhou))] # choice
        print('1P =',state[i])
        istep=check_player(1,i,state)      # choice = gouhouhu ? (player 1)
        switch_list(i,i+istep,state)      # update
    print_list(state)
    gouhou=search_gouhou(2)
    if(len(gouhou)>0):
        i=gouhou[random.randrange(0,len(gouhou))] # choice
        print('2P =',state[i])
        istep=check_player(2,i,state)      # choice = gouhouhu ? (player 1)
        switch_list(i,i+istep,state)      # update
print_list(state)
