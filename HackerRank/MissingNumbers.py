__author__ = 'Michael'
from HackerRank.decorators import get_line
import sys
from collections import defaultdict

sys.stdin = open("inputs/missingNumbers/in1.txt")

input()
a = get_line()
input()
b = get_line()

da = defaultdict(int)
db = defaultdict(int)

for num in a:
    da[num] += 1

for num in b:
    db[num] += 1

for k, v in db.items():
    if da[k] != v:
        print(k, end=' ')