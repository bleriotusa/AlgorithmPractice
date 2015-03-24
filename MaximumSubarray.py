__author__ = 'Michael'

'''
Problem Statement

Given an array A={a1,a2,…,aN} of N elements, find the maximum possible sum of a

Contiguous subarray
Non-contiguous (not necessarily contiguous) subarray.
Empty subarrays/subsequences should not be considered.

Input Format

First line of the input has an integer T. T cases follow.
Each test case begins with an integer N. In the next line, N integers follow representing the elements of array A.

Constraints:

1≤T≤10
1≤N≤105
−104≤ai≤104
The subarray and subsequences you consider should have at least one element.

Output Format

Two, space separated, integers denoting the maximum contiguous and non-contiguous subarray. At least one integer should be selected and put into the subarrays (this may be required in cases where all elements are negative).

'''
import sys
sys.stdin = open('inputs/maxsubarray/in1.txt')

t = int(input())
for i in range(0, t):
    input()
    a = list(map(lambda num: int(num), input().split(' ')))

    non_cont = sum([num for num in a if num > 0] + [0])
    if non_cont == 0:
        non_cont = max(a)

    max_ending_here = 0
    max_so_far = 0
    for x in a:
        max_ending_here = max(0, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)
    if max_so_far == 0:
        max_so_far = max(a)
    print(max_so_far, non_cont)



def max_subarray(arr):
    if not arr:
        return 0
    elif len(arr) == 1:
        return arr[0]
    else:
        half = int(len(arr)/2)
        half1 = max_subarray(arr[:half])
        half2 = max_subarray(arr[half:])
        return max(half1, half2, half1+half2)

print()
l = [-3, -2, 1, 2, 3, -5, 9]
# half = int(len(l)/2)
print(max_subarray(l))
# print(len(l), half)
# print(l[:half], l[half:])