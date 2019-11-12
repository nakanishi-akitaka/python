#!/usr/bin/env python
# -*- coding: utf-8 -*-

# test09-1.py
# class definition
class Person:
    # class parameter
    population = 7400000000
    # constructor
    def __init__(self,Name,Age,Gender,Country):
        self.name    = Name
        self.age     = Age
        self.gender  = Gender
        self.country = Country
    # clas method
    @classmethod
    def belongTo(cls):
        print(cls,'is belong to Human')
        return 'Human'
    # method
    def getName(self):
        print('Name is ', self.name)
        return self.name
    def getAge(self):
        print('Age is ', self.age)
        return self.age
    def getGender(self):
        print('Gender is ', self.gender)
        return self.gender
    def getCountry(self):
        print('Country is ', self.country)
        return self.country

# main routine
m1 = Person('Tarou',39,'male','Japan')
m2 = Person('Alice',28,'female','America')

m1.getName()
m1.getAge()
m1.getGender()
m1.getCountry()

m2.getName()
m2.getAge()
m2.getGender()
m2.getCountry()

Person.belongTo()
print('Population is about',Person.population)
