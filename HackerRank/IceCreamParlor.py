__author__ = 'Michael'
"""
Problem Statement

Sunny and Johnny together have M dollars they want to spend on ice cream. The parlor offers N flavors, and they want to choose two flavors so that they end up spending the whole amount.

You are given the cost of these flavors. The cost of the ith flavor is denoted by ci. You have to display the indices of the two flavors whose sum is M.

Input Format

The first line of the input contains T; T test cases follow.
Each test case follows the format detailed below: The first line contains M. The second line contains N. The third line contains N space-separated integers denoting the price of each flavor. Here, the ith integer denotes ci.

Output Format

Output two integers, each of which is a valid index of a flavor. The lower index must be printed first. Indices are indexed from 1 to N.

Constraints

1≤T≤50
2≤M≤10000
2≤N≤10000
1≤ci ≤10000,where i∈[1,N]
The prices of any two items may be the same and each test case has a unique solution.

Sample Input

2
4
5
1 4 5 3 2
4
4
2 2 4 3
Sample Output

1 4
1 2
Explanation

The sample input has two test cases.
For the 1st, the amount M = 4 and there are 5 flavors at the store. The flavors indexed at 1 and 4 sum up to 4.
For the 2nd test case, the amount M = 4 and the flavors indexed at 1 and 2 sum up to 4.
"""

from HackerRank.decorators import *
import sys
from collections import defaultdict

sys.stdin = open('inputs/icecream/in3.txt')

outfile = open('inputs/icecream/myout3.txt', 'w+')

testcases = int(input())
for i in range(0, testcases):
    m = int(input())
    n = int(input())
    prices = get_line()
    d = dict()
    for i in range(1, len(prices) + 1):
        price = prices[i-1]
        d[price] = i

    for i in range(0, len(prices)):
        if m-prices[i] in d.keys() and i+1 != d[m-prices[i]]:
            print(i+1, d[m-prices[i]])
            # outfile.write("{} {}\n".format(i+1, d[m-prices[i]]))
            break

outfile.close()