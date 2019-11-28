#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### section 18.[1-3],7
class Card(object):
    ''' 
    Represents a standard playing card.
    Spades   -> 3
    Hearts   -> 2
    Diamonds -> 1
    Clubs    -> 0
    Jack  -> 11
    Queen -> 12
    King  -> 13
    '''
    # 18.1
    def __init__(self, suit=0, rank=2):
        self.suit = suit
        self.rank = rank
    # 18.2
    suit_names = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    rank_names = [None, 'Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
    def __str__(self):
        return '%s of %s' % (Card.rank_names[self.rank], Card.suit_names[self.suit])
    # 18.3
    def cmp(self, other):
        # check the suits
        # suits are the same ... check the ranks
        # ranks are the same... it's a tie
        if self.suit > other.suit: return  1
        if self.suit < other.suit: return -1
        if self.rank > other.rank: return  1
        if self.rank < other.rank: return -1
        return 0
    def __lt__(self, other): return self.cmp(other)
    def __le__(self, other): return self.cmp(other)
    def __gt__(self, other): return self.cmp(other)
    def __ge__(self, other): return self.cmp(other)
    def __eq__(self, other): return self.cmp(other)
    def __ne__(self, other): return self.cmp(other)
    # 18.7
    def move_cards(self, hand, num):
        for i in range(num):
            hand.add_card(self.pop_card())
    # train 18.3
    def deal_hands(self, hand, num):
        hand = Hand()
        self.move_cards(hand,num)
# card1 = Card(2, 13)
# card2 = Card(1, 1)
# print(card1>card2)
# print(card1<card2)
# print(card1>=card2)
# print(card1<=card2)
# print(card1==card2)
# print(card1!=card2)
# card1 = Card(1, 1)
# card2 = Card(1, 1)
# print(card1>card2)
# print(card1<card2)
# print(card1>=card2)
# print(card1<=card2)
# print(card1==card2)
# print(card1!=card2)
# card1 = Card(0, 13)
# card2 = Card(1, 1)
# print(card1>card2)
# print(card1<card2)
# print(card1>=card2)
# print(card1<=card2)
# print(card1==card2)
# print(card1!=card2)


#### train 18.1 -> skip

#### section 18.[4-6] & train 18.2
class Deck(object):
    # 18.4
    def __init__(self):
        self.cards = []
        for suit in range(4):
            for rank in range(1,14):
                card = Card(suit, rank)
                self.cards.append(card)
    # 18.5
    def __str__(self):
        res = []
        for card in self.cards:
            res.append(str(card))
        return '\n'.join(res)
    # 18.6
    import random
    def pop_card(self):
        return self.cards.pop()
    def add_card(self, card):
        self.cards.append(card)
    def shuffle(self):
        random.shuffle(self.cards)
    # train 18.2
    def sort(self):
        self.cards.sort()
deck = Deck()
deck.sort()
# print(deck)

#### section 18.7
class Hand(Deck):
    def __init__(self, label=''):
        self.cards = []
        self.label = label
hand = Hand('new hand')
print(hand.cards)
print(hand.label)
deck = Deck()
card = deck.pop_card()
hand.add_card(card)
print(hand)

#### section 18.8
# nothing
#### train 18.3 -> skip
#### train 18.4 -> skip

#### section 18.9
def find_defining_class(obj, meth_name):
    for ty in type(obj).mro():
        if meth_name in ty.__dict__:
            return ty
hand= Hand()
print(find_defining_class(hand, 'shuffle'))
#### section 18.10
# nothing

#### train 18.5

#### section 18.11
# nothing 

#### section 18.12
#### train 18.6
#### train 18.7
