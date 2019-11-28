#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### section 17.1
# nothing

#### section 17.[2-6] & train 17.[1]
class Time:
    '''Represents the time of day .
    attributes: hour, minute, second'''
    def print_time(self):
        print('%.2d:%.2d:%.2d' % (self.hour, self.minute, self.second))
    def time_to_int(self):
        minutes = self.hour * 60 + self.minute
        seconds = minutes * 60 + self.second
        return seconds
    def increment(self, seconds):
        seconds += self.time_to_int()
        return int_to_time(seconds)
    def is_after(self, other):
        return self.time_to_int() > other.time_to_int()
    def __init__(self, hour=0, minute=0, second=0):
        self.hour = hour
        self.minute = minute
        self.second = second
    def __str__(self):
        return '%02d:%02d:%02d' % (self.hour, self.minute, self.second)
    def __add__(self, other):
        if isinstance(other, Time):
            return self.add_time(other)
        else:
            return self.increment(other)
    def __radd__(self, other):
        return self.__add__(other)
    def add_time(self, other):
        seconds = self.time_to_int() + other.time_to_int()
        return int_to_time(seconds)

start = Time()
start.hour = 9
start.minute = 45
start.second = 0
Time.print_time(start)
start.print_time()

#### train 17.1
print(start.time_to_int())

#### section 17.3
def int_to_time(seconds):
    time = Time()
    minutes, time.second = divmod(seconds,60)
    time.hour, time.minute = divmod(minutes,60)
    return time
end=start.increment(900)
end.print_time()

#### section 17.4
print(end.is_after(start))

#### section 17.5
time = Time()
time.print_time()
time = Time(9)
time.print_time()
time = Time(9,45)
time.print_time()
time = Time(9,45,13)
time.print_time()

#### train 17.[2-5]
class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def __str__(self):
        return '(%.1f, %.1f)' % (self.x, self.y)
    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x+other.x,self.y+other.y)
        elif isinstance(other, tuple):
            return self.move(other)
        else:
            return self
    def __radd__(self, other):
        return self.__add__(other)
    def move(self, dv):
        return Point(self.x+dv[0],self.y+dv[1])
p = Point()
print(p.x,p.y)
p = Point(1)
print(p.x,p.y)
p = Point(2,3)
print(p.x,p.y)

#### section 17.6
time = Time(9,45)
print(time)

#### train 17.3
p = Point(3,4)
print(p)

#### section 17.7
start = Time(9, 45)
duration = Time(1, 35)
print(start+duration)

#### train 17.4
p = Point(3,4)
q = Point(10,20)
print(p+q)

#### section 17.8
start = Time(9, 45)
duration = Time(1, 35)
print(start+duration)
print(start+duration.time_to_int())
print(start+5700)
print(5700+start)

#### train 17.5
p = Point(3,4)
print(p+(10,20))
print((10,20)+p)

#### section 17.8 (typo 10.9?)
t1 = Time(7,43)
t2 = Time(7,41)
t3 = Time(7,37)
total = sum((t1,t2,t3))
print(total)
p1 = Point(1,4)
p2 = Point(2,5)
p3 = Point(3,6)
total = sum((p1,p2,p3))
print(total)

#### section 17.10
p = Point(3, 4)
print(p.__dict__)
def print_attributes(obj):
    for attr in obj.__dict__:
        print(attr,getattr(obj,attr))
print_attributes(p)

#### section 17.11
# nothing

#### train 17.6 -> skip

#### section 17.12
# nothing

#### section 17.13
#### train 17.7
# BadKangaroo.py
class BadKangaroo(object):
    def __init__(self, contents=[]):
        self.pouch_contents = contents
    def __str__(self):
        t = [ object.__str__(self) + ' with pouch contents:']
        for obj in self.pouch_contents:
            s = '    ' + object.__str__(obj)
            t.append(s)
        return '\n'.join(t)
    def put_in_pouch(self, item):
        self.pouch_contents.append(item)
kanga = BadKangaroo()
roo = BadKangaroo()
kanga.put_in_pouch('wallet')
kanga.put_in_pouch('car keys')
kanga.put_in_pouch(roo)
print(kanga)
print(roo)


class GoodKangaroo(object):
    def __init__(self, contents=None):
        if contents == None:
            contents = []
        self.pouch_contents = contents
    def __str__(self):
        t = [ object.__str__(self) + ' with pouch contents:']
        for obj in self.pouch_contents:
            s = '    ' + object.__str__(obj)
            t.append(s)
        return '\n'.join(t)
    def put_in_pouch(self, item):
        self.pouch_contents.append(item)
kanga = GoodKangaroo()
roo = GoodKangaroo()
kanga.put_in_pouch('wallet')
kanga.put_in_pouch('car keys')
kanga.put_in_pouch(roo)
print(kanga)
print(roo)

class Kangaroo(object):
    def __init__(self, contents=None):
        if contents == None:
            contents = []
        self.pouch_contents = contents
    def __str__(self):
        t = [ object.__str__(self) + ' with pouch contents:']
        return '\n'.join(t)
    def put_in_pouch(self, item):
        self.pouch_contents.append(item)
kanga = Kangaroo()
roo = Kangaroo()
kanga.put_in_pouch('wallet')
kanga.put_in_pouch('car keys')
kanga.put_in_pouch(roo)
print(kanga)
print(roo)

#### train 17.8 -> skip
