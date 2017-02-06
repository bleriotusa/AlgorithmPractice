"""
https://www.hackerrank.com/challenges/ctci-making-anagrams/forum
Alice is taking a cryptography class and finding anagrams to be very useful. We consider two strings to be anagrams of each other if the first string's letters can be rearranged to form the second string. In other words, both strings must contain the same exact letters in the same exact frequency For example, bacdc and dcbac are anagrams, but bacdc and dcbad are not.

Alice decides on an encryption scheme involving two large strings where encryption is dependent on the minimum number of character deletions required to make the two strings anagrams. Can you help her find this number?

Given two strings,  and , that may or may not be of the same length, determine the minimum number of character deletions required to make  and  anagrams. Any characters can be deleted from either of the strings.

This challenge is also available in the following translations:

Chinese
Russian
Input Format

The first line contains a single string, .
The second line contains a single string, .

Constraints

It is guaranteed that  and  consist of lowercase English alphabetic letters (i.e.,  through ).
Output Format

Print a single integer denoting the number of characters you must delete to make the two strings anagrams of each other.

Sample Input

cde
abc
Sample Output

4
Explanation

We delete the following characters from our two strings to turn them into anagrams of each other:

Remove d and e from cde to get c.
Remove a and b from abc to get c.
We must delete  characters to make both strings anagrams, so we print  on a new line.
"""

from unittest import TestCase
from collections import defaultdict

import sys


def number_needed(a, b):
    mapping_a = defaultdict(int)
    mapping_b = defaultdict(int)
    for char in a:
        mapping_a[char] += 1

    for char in b:
        mapping_b[char] +=1

    num_needed = 0
    # print(set(mapping_a.keys()), set(mapping_b.keys()))
    for char in set(mapping_a.keys()).union(set(mapping_b.keys())):
        num_needed += abs(mapping_a[char] - mapping_b[char])

    return num_needed





class TestMakingAnagrams(TestCase):
    def setup(self):
        self.a = input().strip()
        self.b = input().strip()

    def test_1(self):
        sys.stdin = open('inputs/MakingAnagrams/in1.txt')
        self.setup()
        self.assertEqual(number_needed(self.a, self.b), 4)
