#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import sum
import random

game=0
turn=1
state=[0,1,2,3,4,5,6,7,8,9]
# state=[0,0,0,0,0,0,0,0,0,9]

def check_win(turn):
    global game
    if(sum(state)==0):
        print(turn,'win')
        game=100

def ai():
    global state
    for i in range(len(state)):
        if(state[i]!=0):
            choice=i
            break
    return choice

while game < 1:
    if(turn%2==0):
        # my turn
        print(state)
#       n=random.randrange(1,10)
        n=ai()
        print('  my choice number = ',n)
        state[n]=0
        check_win('I')
        turn+=1
    else:
        # your turn
        print(state)
        n=int(input('your choice number = '))
        state[n]=0
        check_win('you')
        turn+=1
