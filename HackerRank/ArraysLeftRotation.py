"""
A left rotation operation on an array of size  shifts each of the array's elements  unit to the left. For example, if left rotations are performed on array , then the array would become .

Given an array of  integers and a number, , perform  left rotations on the array. Then print the updated array as a single line of space-separated integers.

Input Format

The first line contains two space-separated integers denoting the respective values of  (the number of integers) and  (the number of left rotations you must perform).
The second line contains  space-separated integers describing the respective elements of the array's initial state.

Constraints




Output Format

Print a single line of  space-separated integers denoting the final state of the array after performing  left rotations.

Sample Input

5 4
1 2 3 4 5
Sample Output

5 1 2 3 4
Explanation

When we perform  left rotations, the array undergoes the following sequence of changes:


Thus, we print the array's final state as a single line of space-separated values, which is 5 1 2 3 4.
https://www.hackerrank.com/challenges/ctci-array-left-rotation
"""
import sys
from unittest import TestCase


def rotate_left(a, n, k):
    """ 1. Store the first variable
        2. Go through the array and move next element to current
        3. When there is no next, just insert the stored first variable

     """
    if not a:
        return a

    first = a[0]
    for i in range(0, len(a)):
        if i + 1 < len(a):
            a[i] = a[i + 1]
        else:
            a[i] = first


def array_left_rotation(a, n, k):
    for i in range(0, k):
        rotate_left(a, n, k)

    return a


class TestArrayLeftRotation(TestCase):
    def setup(self):
        self.n, self.k = map(int, input().strip().split(' '))
        self.a = list(map(int, input().strip().split(' ')))

    def test_1(self):
        sys.stdin = open('inputs/arrayLeftRotation/in1.txt')
        self.setup()
        answer = array_left_rotation(self.a, self.n, self.k)
        print(*answer, sep=' ')
        self.assertListEqual(answer, [5, 1, 2, 3, 4])

    def test_0(self):
        sys.stdin = open('inputs/arrayLeftRotation/in2.txt')
        self.setup()
        answer = array_left_rotation(self.a, self.n, self.k)
        print(*answer, sep=' ')
        self.assertListEqual(answer, [5])
