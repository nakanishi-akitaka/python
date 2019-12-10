#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Fibonacci numbers module
def fib(n):
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a+b
    print()

def fib2(n):
    result = []
    a, b = 0, 1
    while b < n:
        result.append(b)
        a, b = b, a+b
    return result

if __name__ == "__main__":
    fib(50)
#   import sys
#   fib(int(sys.argv[1]))
