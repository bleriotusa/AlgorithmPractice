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


def array_left_rotation(a, n, k):
    """ 1. For each number,
        2. Calculate the final destination after k rotations.
        3. Add the number to the new destination in a new array
     """
    if not a:
        return a

    b = [0] * len(a)

    for i in range(0, n):
        b[i - k % n] = a[i]

    return b


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

    def test_3(self):
        sys.stdin = open('inputs/arrayLeftRotation/in3.txt')
        self.setup()
        sys.stdin = open('inputs/arrayLeftRotation/out3.txt')
        answer_key = list(map(int, input().strip().split(' ')))
        answer = array_left_rotation(self.a, self.n, self.k)
        print(*answer, sep=' ')
        self.assertListEqual(answer, answer_key)


