#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### section 9.1
####fin=open('words.txt')
####print(fin)
####print(fin.readline())
####print(fin.readline())
####line=fin.readline()
####word=line.strip()
####print(word)
####fin=open('words.txt')
####for line in fin:
####    word = line.strip()
####    print(word)

#### train 9.1
####fin=open('words.txt')
####for line in fin:
####    word = line.strip()
####    if(len(word)>=20):
####        print(word)

#### section 9.2
####def has_no_e(word):
####    for i in word:
####        if(i=='e'):
####            return False
####    return True
####print(has_no_e('Hello'))
####print(has_no_e('world'))
####def has_no_e(word):
####    for i in word:
####        if(i=='e'):
####            return 
####    print(word)
####fin=open('words.txt')
####for line in fin:
####    word = line.strip()
####    has_no_e(word)

#### train 9.3
####def avoids(word,letter):
####    for i in word:
####        if(i==letter):
####            return False
####    return True

#### train 9.4
####def uses_only(word,letters):
####    for i in word:
####        luse=False
####        for j in letters:
####            if(i==j):
####                luse=True
####        if(luse==False):
####            return False
####    return True
####print(uses_only('hoealfalfa','acefhlo'))

#### train 9.5
####def uses_all(word,letters):
####    for i in letters:
####        luse=False
####        for j in word:
####            if(i==j):
####                luse=True
####        if(luse==False):
####            return False
####    return True
####print(uses_all('hoealfalfa','acefhlo'))

#### train 9.6
####def is_abecedarian(word):
####    ord1=ord(word[0])
####    for i in word[1:]:
####        ord2=ord(i)
####        if(ord1>ord2):
####            return False
####        ord1=ord2
####    return True
####print(is_abecedarian('abcdefg'))

#### section 9.3
####def uses_only(word,letters):
####    for i in word:
####        if i not in letters :
####            return False
####    return True
####print(uses_only('hoealfalfa','acefhlo'))
####
####def uses_all(word,letters):
####    for i in letters:
####        if i not in word :
####            return False
####    return True
####print(uses_all('hoealfalfa','acefhlo'))

#### section 9.4
####def is_abecedarian(word):
####    pre=word[0]
####    for i in word:
####        if i < pre:
####            return False
####        pre = i
####    return True
####print(is_abecedarian('abc'))
####print(is_abecedarian('abca'))
####def is_abecedarian(word):
####    if len(word) <= 1:
####        return True
####    if word[0] > word[1]:
####        return False
####    return is_abecedarian(word[1:])
####print(is_abecedarian('abc'))
####print(is_abecedarian('abca'))
####def is_abecedarian(word):
####    i=0
####    while i < len(word)-1:
####        if word[i+1] < word[i]:
####            return False
####        i+=1
####    return True
####print(is_abecedarian('abc'))
####print(is_abecedarian('abca'))

#### section 9.9 (typo? to be 9.5)
#### nothing

#### train 9.7
def cartalk(word):
    if(len(word)<6):
        return
    i=0
    c=0
    while i < len(word)-1:
        if word[i] == word[i+1]:
            c +=1
            if c==3:
                return True
            i+=2
        else:
            c=0
            i+=1
    return False
    print(word)

fin=open('words.txt')
for line in fin:
    word = line.strip()
    if(cartalk(word)):
        print(word)
#### train 9.8 & 9.9 -> skip
