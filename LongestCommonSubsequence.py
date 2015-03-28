__author__ = 'Michael'
"""
https://www.hackerrank.com/challenges/dynamic-programming-classics-the-longest-common-subsequence
Problem Statement

A subsequence is a sequence that can be derived from another sequence by deleting some elements without changing the order of the remaining elements. Longest common subsequence (LCS) of 2 sequences is a subsequence, with maximal length, which is common to both the sequences.

Given two sequence of integers, A=[a1,a2,…,an] and B=[b1,b2,…,bm], find any one longest common subsequence.

In case multiple solutions exist, print any of them. It is guaranteed that at least one non-empty common subsequence will exist.

Input Format

First line contains two space separated integers, n and m, where n is the size of sequence A, while m is size of sequence B. In next line there are n space separated integers representing sequence A, and in third line there are m space separated integers representing sequence B.

n m
A1 A2 … An
B1 B2 … Bm
Constraints

1≤n≤100
1≤m≤100
0≤ai<1000,where i∈[1,n]
0≤bj<1000,where j∈[1,m]

Output Format

Print the longest common subsequence and each element should be separated by at least one white-space. In case of multiple answers, print any one of them.

Sample Input

5 6
1 2 3 4 1
3 4 1 2 1 3
Sample Output

1 2 3
Explanation

There is no common subsequence with length larger than 3. And "1 2 3", "1 2 1", "3 4 1" are all correct answers.

"""

from decorators import *

@Track_Calls
@memoize
def lcs(a: [int], b: [int]):
    if not a or not b:
        return []

    else:
        # case1: first a element is in the answer
        if a[-1] == b[-1]:
            return lcs(a[:-1], b[:-1]) + [a[-1]]
        else:
            case1 = lcs(a[:-1], b)
            case2 = lcs(a, b[:-1])
            return case1 if len(case1) > len(case2) else case2


if __name__ == '__main__':
    sampleA = tuple(map(lambda x: int(x), '1 2 3 4 1'.split(' ')))
    sampleB = tuple(map(lambda x: int(x), '3 4 1 2 1 3'.split(' ')))
    print(sampleA, sampleB)
    print(lcs(sampleA, sampleB))
    print(lcs.calls)