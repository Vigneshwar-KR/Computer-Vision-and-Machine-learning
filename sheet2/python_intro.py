#!/usr/bin/env python3
# -*- coding: utf-8 -*-



print("Introduction to Python programming")



#
# Task 1
#
# TODO: Create a list with values 1--20.
list = [(i + 1) for i in range(20)] # Task 1.1
print(list)

# TODO: Use a list comprehension to square all odd values in this list (keep even values unchanged).
list = [list[i] * list[i] if i % 2 == 0 else list[i] for i in range(20)] # Task 1.2 (if else put infront of the loop) (only if after)
print(list)

# TODO: Request a number from the user (terminal keyboard input), loop this to get 4 numbers, and sort the numbers in ascending order.
num = []
for i in range(4): # Task 1.3
    num.append(int(input("Enter number :")))
num.sort()
print(num)



#
# Task 2
#
# Write a function that ...
#
# TODO: ... squares all elements of a list.
def square_list(a):
    size = len(a)
    for i in range(size):
        a[i] *= a[i]

a = [1,2,3,4,5,6,7,8,9,10]
square_list(a)
print("result of Task 2.1:")
print('We will square the list [1,2,3,4,5,6,7,8,9,10].')
print(a)

# TODO: ... recursively calculates the sum of all elements in a list.
def recursive_sum(a, count):
    if count <= 0:
        return 0
    else:
        return a[count - 1] + recursive_sum(a , count = count - 1)

a = [1,2,3,4,5,6,7,8,9,10]
size = len(a)
print("result of Task 2.2:")
print('We will sum the list [1,2,3,4,5,6,7,8,9,10].')
print(recursive_sum(a,size))

# TODO: ... uses the built-in Python function `sum(list)` to calculate the arithmetic mean of all elements in a list.
print("result of Task 2.3:")
print('We will sum the list [1,2,3,4,5,6,7,8,9,10].')
print(sum(a))



#
# Task 3
#

import math

# Write a class `Vec2` that has ...
# TODO: ... a variable `id`, and a "global" class variable `gid` that is used to assign each instance a unique `id`.
gid = 0

class Vec():
    # TODO: ... a constructor that initializes two variables `x` and `y`.
    def __init__(self, x, y):
        global gid
        self.x = x
        self.y = y
        self.id = gid
        gid += 1

    # TODO: ... a member function `add(self, rhs)` that calculates the component-wise sum of the vector and another one (`rhs`).
    def length(self):
        length = math.sqrt(self.x ** 2 + self.y ** 2)
        return length
    def add(self, rhs):
        return Vec(self.x + rhs.x, self.y + rhs.y)

# TODO: Vec2 demo
x = Vec(1,1)
print(x.id)
print(x.length())
y = Vec(-1,-1)
print(y.id)
z = x.add(y)
print(z.length())
