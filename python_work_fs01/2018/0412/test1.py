#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### section 15.1
class Point(object):
    '''Represents a point in 2-D space.  '''
print(Point)
blank=Point()
print(blank)

#### section 15.2
blank.x=3.0
blank.y=4.0
print(blank.x)
print(blank.y)
print('(%g, %g)' % (blank.x, blank.y))
import math
distance = math.sqrt(blank.x**2+blank.y**2)
print(distance)
def print_point(p):
    print('(%g, %g)' % (p.x, p.y))
print_point(blank)

#### train 15.1
def distance_between_points(p,q):
    return math.sqrt((p.x-q.x)**2+(p.y-q.y)**2)
a=Point()
b=Point()
a.x=0.0
a.y=0.0
b.x=3.0
b.y=4.0
print(distance_between_points(a,b))

#### section 15.3
class Rectangle(object):
    '''
    Represents a rectangle
    attributes: width, height, corner
    '''

box=Rectangle()
box.width = 100.0
box.height = 200.0
box.corner = Point()
box.corner.x = 0.0
box.corner.y = 0.0

#### section 15.4
print('section 15.4')
def find_center(rect):
    p = Point()
    p.x = rect.corner.x + rect.width/2.0
    p.y = rect.corner.y + rect.height/2.0
    return p
center = find_center(box)
print_point(center)

#### section 15.5
print('section 15.5')
box.width = box.width + 50
box.height = box.height + 50
def grow_rectangle(rect, dwidth, dheight):
    rect.width += dwidth
    rect.height += dheight
print(box.width,box.height)
grow_rectangle(box,50,100)
print(box.width,box.height)

#### train 15.2
print('train 15.2')
def move_rectangle(rect, dx, dy):
    rect.corner.x += dx
    rect.corner.y += dy
print_point(find_center(box))
move_rectangle(box,50,100)
print_point(find_center(box))

#### section 15.6
print('section 15.6')
p1 = Point()
p1.x = 3.0
p1.y = 4.0
import copy
p2 = copy.copy(p1)
print_point(p1)
print_point(p2)
print(p1 is p2)
print(p1 == p2)
print('shallow copy, box2')
box2 = copy.copy(box)
print(box is box2)
print(box == box2)
print(box.corner is box2.corner)
print(box.corner == box2.corner)
print('deep copy, box3')
box3 = copy.deepcopy(box)
print(box is box3)
print(box == box3)
print(box.corner is box3.corner)
print(box.corner == box3.corner)

#### train 15.3
print('train 15.3')
def move_rectangle2(rect, dx, dy):
    rect2 = copy.deepcopy(rect)
    rect2.corner.x += dx
    rect2.corner.y += dy
    return rect2
print_point(find_center(box))
box2=move_rectangle2(box,50,100)
print_point(find_center(box2))
print_point(find_center(box))

#### section 15.7
print('section 15.7')
p = Point()
print(hasattr(p,'x'))
print(hasattr(p,'y'))
print(hasattr(p,'z'))

p.x=3.0
p.y=4.0
p.z=5.0
print(hasattr(p,'x'))
print(hasattr(p,'y'))
print(hasattr(p,'z'))
p.jugemujugemugokounosurikire='kaijarisuigyo'
print(hasattr(p,'jugemujugemugokounosurikire'))

#### section 15.8
# nothing 

#### section 15.9
#### train 15.4 -> skip

#### section 15.7
#### train 15.3

#### section 15.8
# nothing 

#### section 15.9
#### train 15.4 -> skip
