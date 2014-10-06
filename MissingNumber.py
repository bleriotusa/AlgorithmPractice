"""
Missing number
Given an array containing all numbers from 1 to N with the exception of one print the missing number to the standard output.
Example input:
array: 5 4 1 2
Example output:
3
Note: This challenge was part of Microsoft interviews. The expected complexity is O(N).
"""

def find_missing_number(v):
    # Write your code here
    # To print results to the standard output you can use print
    # Example: print "Hello world!"
    maxx = max(v)
    checklist = [True]
    for i in range(1, maxx+1):
        checklist.append(False)
    for num in v:
        checklist[num] = True
    for i in range(1, maxx+1):
        if not checklist[i]:
            print i