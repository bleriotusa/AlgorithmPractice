__author__ = 'Michael'
"""
Sqrt
Given an integer number N, compute its square root without using any math library functions and print the result to
standard output.
Please round the result downwards to the nearest integer (e.g both 7.1 and 7.9 are rounded to 7)
Expected complexity: O(logN), O(1)
Example input:
N: 17
Example output:
4
"""


def compute_sqrt(n):
    # Write your code here
    # To print results to the standard output you can use print
    # Example: print "Hello world!"
    answers = range(1, n + 1)
    print(find_sqrt(answers, n))


def find_sqrt(answers, n):
    # print(answers)
    middle = answers[int(len(answers) / 2)]
    # print(middle, middle ** 2)
    if middle ** 2 == n or len(answers) == 1:
        return middle
    elif middle ** 2 < n:
        return find_sqrt(answers[int(len(answers) / 2):], n)
    elif middle ** 2 > n:
        return find_sqrt(answers[:int(len(answers) / 2)], n)


compute_sqrt(17)