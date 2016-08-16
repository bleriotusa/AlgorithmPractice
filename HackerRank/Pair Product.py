__author__ = 'Michael'
'''
Write to the standard output the greatest product of 2 numbers to be divisible by 3 from a given array of pozitive integers.
Example input:
6, 8, 8, 7, 2, 5
Example output:
48
'''

def max_prod(v):
    # Write your code here
    # To print results to the standard output you can use print
    # Example: print "Hello world!"
    max = 0
    for i in range(0, len(v)):
        for p in range(0, len(v)):
            if i is not p:
                temp = (v[i]*v[p])  if (v[i]*v[p]) % 3 == 0 else max
                max = temp if temp > max else max
    print(max)

def max_prod2(v):
    temp = [v[i]*v[g] for i in range(0, len(v)) for g in range(0, len(v)) if i != g and v[i]*v[g] % 3 == 0]
    print(max(temp))

test_list = [6, 8, 8, 7, 2, 5]
max_prod(test_list )
max_prod2(test_list )