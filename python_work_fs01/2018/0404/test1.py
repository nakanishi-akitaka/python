#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### section 8.1
####fruit='banana'
####print(fruit[0])

#### section 8.2
####fruit='banana'
####print(fruit[len(fruit)-1])
####print(fruit[-1])

#### section 8.3
####fruit='banana'
####i=0
####while i < len(fruit):
####    letter = fruit[i]
####    print(letter)
####    i+=1

#### train 8.1
####fruit='banana'
####def rev(letter):
####    i=len(letter)-1
####    while i > -1:
####        print(fruit[i])
####        i-=1
####rev(fruit)

####fruit='banana'
####for c in fruit:
####    print(c)

#### train 8.2
####prefixes='JKLMNOPQ'
####sufix = 'ack'
####for l in prefixes:
####    if(l=='O' or l=='Q'):
####        print(l+'u'+sufix)
####    else:
####        print(l+sufix)

#### section 8.4
####a='Monty Python'
####print(a[0:5])
####fruit='banana'
####print(fruit[:3])
####print(fruit[3:])
####print(fruit[3:3])
####print(fruit[:])

#### section 8.5
####greeting='Hello world!'
####new_greeting='J'+greeting[1:]
####print(new_greeting)

#### section 8.6 
####def find(word, letter):
####    i = 0
####    while i < len(word):
####        if word[i] == letter:
####            return i
####        i += 1
####    return -1
####print(find('Hello World!','l'))

#### train 8.4
####def find2(word, letter, j):
####    i = j
####    while i < len(word):
####        if word[i] == letter:
####            return i
####        i += 1
####    return -1
####print(find2('Hello World!','l',4))

#### section 8.7
####word = 'banana'
####count = 0
####for l in word:
####    if l == 'a':
####        count += 1
####print(count)

#### train 8.5
####word = 'banana'
####letter='a'
####def counter(word,letter):
####    c = 0
####    for l in word:
####        if l == letter:
####            c += 1
####    return c
####print(counter(word,letter))


#### train 8.6
####def find2(word, letter, j):
####    i = j
####    while i < len(word):
####        if word[i] == letter:
####            return i
####        i += 1
####    return -1
####def counter2(word,letter):
####    c = 1
####    for i in range(len(word)-1):
####        if(find2(word,letter,i)==i):
####            c+=1
####    return c
####word = 'banana'
####letter='a'
####print(counter2(word,letter))

#### section 8.8
####word='banana'
####print(word.upper())
####print(word.find('a'))
####print(word.find('na'))
####print(word.find('na',3))
####print(word.find('b',1,5))

#### train 8.7
####word='banana'
####print(word.count('a'))
####print(word.count('a',2))
####print(word.count('a',2,4))

#### train 8.8
####word='banana'
####print(word.strip())
####print(word.strip('a'))
####print(word.strip('bn'))
####print(word.replace('a','e'))
####print(word.replace('a','e',2))

#### section 8.9
####print('a' in 'banana')
####print('seed' in 'banana')
####def in_both(w1,w2):
####    for l in w1:
####        if l in w2:
####            print(l)
####in_both('apples','oranges')

#### section 8.10
####w='banana'
####if w == 'banana':
####    print('All right, bananas.')
####w='Pineapple'
####if w < 'banana':
####    print('Your word, '+w+', comes before banana.')
####elif w > 'banana':
####    print('Your word, '+w+', comes before banana.')
####else:
####    print('All right, bananas.')

#### section 8.11 & train8.9
####def is_reverse(w1,w2):
####    if len(w1) != len(w2):
####        return False
####    i = 0
####    j = len(w2)-1
####    while j>=0:
####        if w1[i] != w2[j]:
####            return False
####        i+=1
####        j-=1
####    return True
####print(is_reverse('pots','stop'))

#### train8.10
####fruit='banana'
####print(fruit[0:5:2])
####print(fruit[::-1])
####def is_palindrome(word):
####    return word==word[::-1]
####print(is_palindrome('banana'))
####print(is_palindrome('ana'))

#### train8.11 -> skip
#### train8.12
#### www.greenteapress.com/thinkpython/code/rotate.py
####abc='abcdefghijklmnopqrstuvwxyz'
####abc='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
####for l in abc:
####    print(ord(l))
def rotate_letter(letter,n):
    if letter.isupper():
        start = ord('A')
    elif letter.islower():
        start = ord('a')
    else:
        return letter
    c = ord(letter) - start
    i = (c + n)%26 + start
    return chr(i)
def rotate_word(word,n):
    res = ''
    for letter in word:
        res += rotate_letter(letter,n)
    return res
print(rotate_word('cheer',7))
print(rotate_word('melon',-10))
print(rotate_word('sleep',9))
