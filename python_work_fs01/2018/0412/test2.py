#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### section 16.1
class Time:
    '''Represents the time of day .
    attributes: hour, minute, second'''

time1 = Time()
time1.hour = 11
time1.minute = 59
time1.second = 30

#### train 16.1
def print_time(t):
    print('%.2d:%.2d:%.2d' % (t.hour, t.minute, t.second))
print_time(time1)

#### train 16.2
def is_after(t1,t2):
    print((t1.hour > t2.hour)
    or (t1.hour == t2.hour and t1.minute > t2.minute)
    or (t1.hour == t2.hour and t1.minute == t2.minute and t1.second > t2.second))
time2 = Time()
time2.hour = 11
time2.minute = 59
time2.second = 29
print_time(time2)
is_after(time1,time2)

#### section 16.2
print('#### section 16.2')
def add_time(t1,t2):
    sum = Time()
    sum.hour = t1.hour + t2.hour
    sum.minute = t1.minute + t2.minute
    sum.second = t1.second + t2.second
    if sum.second >=60:
        sum.second -= 60
        sum.minute +=1
    if sum.minute >=60:
        sum.minute -= 60
        sum.hour +=1
    return sum
start = Time()
start.hour = 9
start.minute = 45
start.second = 0
end = Time()
end.hour = 1
end.minute = 35
end.second = 0
print_time(add_time(start,end))

#### section 16.3
print('#### train 16.3')
def increment(time, seconds):
    time.second += seconds
    if time.second >=60:
        time.second -= 60
        time.minute +=1
    if time.minute >=60:
        time.minute -= 60
        time.hour +=1
print_time(start)
increment(start,360)
print_time(start)

#### train 16.3
print('#### train 16.3')
def increment2(time, seconds):
    time.second += seconds
    if time.second // 60 > 1:
        time.minute += time.second // 60
        time.second  = time.second %  60
    if time.minute >=60:
        time.hour   += time.minute // 60
        time.minute  = time.minute %  60
start = Time()
start.hour = 9
start.minute = 45
start.second = 0
print_time(start)
increment2(start,4000)
print_time(start)

#### train 16.4
print('#### train 16.4')
import copy
def increment3(time, seconds):
    time2 = copy.deepcopy(time)
    time2.second += seconds
    if time2.second // 60 > 1:
        time2.minute += time2.second // 60
        time2.second  = time2.second %  60
    if time2.minute >=60:
        time2.hour   += time2.minute // 60
        time2.minute  = time2.minute %  60
    return time2
start = Time()
start.hour = 9
start.minute = 45
start.second = 0
print_time(start)
start2=increment3(start,4000)
print_time(start2)

#### section 16.4
print('#### section 16.4')
def time_to_int(time):
    minutes = time.hour * 60 + time.minute
    seconds = minutes * 60 + time.second
    return seconds
def int_to_time(seconds):
    time = Time()
    minutes, time.second = divmod(seconds,60)
    time.hour, time.minute = divmod(minutes,60)
    return time
time1 = Time()
time1.hour = 12
time1.minute = 34
time1.second = 56
time2=int_to_time(time_to_int(time1))
print_time(time1)
print_time(time2)
int1=3600*12+60*34+56
int2=time_to_int(int_to_time(int1))
print(int1,int2)

def add_time2(t1,t2):
    seconds = time_to_int(t1) + time_to_int(t2)
    return int_to_time(seconds)
start = Time()
start.hour = 9
start.minute = 45
start.second = 0
end = Time()
end.hour = 1
end.minute = 35
end.second = 0
print_time(add_time2(start,end))

#### train 16.5
print('#### train 16.5')
def increment4(t1, seconds):
    seconds2 = time_to_int(t1) + seconds
    return int_to_time(seconds2)
start = Time()
start.hour = 9
start.minute = 45
start.second = 0
print_time(start)
start2=increment4(start,4000)
print_time(start2)

#### section 16.5
def valid_time(time):
    if time.hour < 0 or time. minute < 0 or time. second < 0:
        return False
    if time.minute >= 60 or time.second >= 60:
        return False
    return True

def add_time3(t1,t2):
    if not valid_time(t1) or not valid_time(t2):
        raise ValueError('invalid time object in add_time3')
    seconds = time_to_int(t1) + time_to_int(t2)
    return int_to_time(seconds)

def add_time4(t1,t2):
    assert valid_time(t1) and valid_time(t2), 'invalid time object in add_time3'
    seconds = time_to_int(t1) + time_to_int(t2)
    return int_to_time(seconds)
start = Time()
start.hour = 9
start.minute = 45
start.second = 0
end = Time()
end.hour = 1
end.minute =-35
end.second = 0
# print_time(add_time3(start,end))
print_time(add_time4(start,end))

#### section 16.6
# nothing

#### section 16.7
#### train 16.6
#### train 16.7
