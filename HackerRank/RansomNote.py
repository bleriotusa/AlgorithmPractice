"""
A kidnapper wrote a ransom note but is worried it will be traced back to him. He found a magazine and wants to know if he can cut out whole words from it and use them to create an untraceable replica of his ransom note. The words in his note are case-sensitive and he must use whole words available in the magazine, meaning he cannot use substrings or concatenation to create the words he needs.

Given the words in the magazine and the words in the ransom note, print Yes if he can replicate his ransom note exactly using whole words from the magazine; otherwise, print No.

Input Format

The first line contains two space-separated integers describing the respective values of  (the number of words in the magazine) and  (the number of words in the ransom note).
The second line contains  space-separated strings denoting the words present in the magazine.
The third line contains  space-separated strings denoting the words present in the ransom note.
"""
from collections import defaultdict
from unittest import TestCase

import sys

'''
ransom_note returns true if the set of words M in the magazine is a superset of the set of words R in ransom
'''
def ransom_note(magazine, ransom):
    if not set(magazine).issuperset(ransom):
        return False

    magCounts = defaultdict(int)
    ransomCounts = defaultdict(int)

    for word in magazine:
        magCounts[word] += 1

    for word in ransom:
        ransomCounts[word] += 1

    for word in ransomCounts.keys():
        if magCounts[word] < ransomCounts[word]:
            return False

    return True



class TestRansomNote(TestCase):
    def setup(self):
        m, n = map(int, input().strip().split(' '))
        self.magazine = input().strip().split(' ')
        self.ransom = input().strip().split(' ')

    def test_1(self):
        sys.stdin = open('inputs/ransomNote/in1.txt')
        self.setup()
        answer = ransom_note(self.magazine, self.ransom)
        self.assertEqual(answer, True)

    def test_2(self):
        sys.stdin = open('inputs/ransomNote/in2.txt')
        self.setup()
        answer = ransom_note(self.magazine, self.ransom)
        self.assertEqual(answer, False)

    def test_3(self):
        sys.stdin = open('inputs/ransomNote/in3.txt')
        self.setup()
        answer = ransom_note(self.magazine, self.ransom)
        self.assertEqual(answer, False)