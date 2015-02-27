'''
You are given a list of integer numbers [a1, a2, ..., an] and are required to find the
subarray with the maximum sum that doesn't contain consecutive elements.
Expected complexity: O(N)
Example input:
[2, 5, 6, 5, 3]
Example output:
11
Explanation:
2 + 6 + 3
'''


def find_max_sum(v):
    # Write your code here
    # To print results to the standard output you can use print
    # Example: print "Hello world!"
    if v == []:
        return 0
    else:
        maximum = 0

        for i in range(0, len(v)):
            bottom = find_max_sum(v[0:i])
            top = find_max_sum(v[i + 1:])
            temp = bottom + v[i] + top
            if temp > maximum and (i == 0 or v[i-1] + 1 != v[i] and v[i-1] != v[i]) and (i == len(v)-1 or v[i+1] - 1 != v[i] and v[i+1] != v[i]):
                print(bottom, top, v[i], temp)
                maximum = temp
        return maximum

from functools import reduce
def find_max_sum2(v):
    d = dict()
    for i in range(0, len(v)):
        d[v[i]] = i
    subsets = []
    for i in range(0, len(v)):
        for j in range(i, len(v)):
            subsets.append([x for x in v[i:j+1]])
    for i in range(0, len(v)):
        for j in range(i, len(v)):
            subsets.append({x for x in v[0:j+1] if x is not j})
    no_consecs = filter(lambda x: not_consecutive(x), subsets)
    print(no_consecs)
    # mapped = map(lambda x: sum(x), no_consecs)
    # print(list(mapped))

    for subset in subsets: print(subset)
    print('')
    for subset in no_consecs:
        print(subset)

def not_consecutive(s, key=lambda x: x):
    if not s:
        return False
    elif len(s) == 1:
        return False
    else:
        s.sort(key=key)
        # print(s[1])
        if type(key(s[1])) == tuple:
            return False
        return (key(s[1])-key(s[0]))**2 > 1 or not_consecutive(s[1:])
print(not_consecutive([1,2,3]))
print(not_consecutive([1,3,5]))
print(not_consecutive([2, 5, 1, 3, 4]))


def find_max(v):
    print(list(enumerate(v)))
    return find_max_sum3(list(enumerate(v)))


def find_max_sum3(v):
    if not v:
        return []
    # if len(v) == 2 and (v[1][0] - v[0][0])**2 <=1:
    #     return [v[1]] if v[1][1] > v[0][1] else [v[0]]
    else:
        l = []
        for i in range(0, len(v)):
            sub1 = find_max_sum3(v[0:i])
            sub_mid = [v[i]]
            sub2 = find_max_sum3(v[i+1:])
            sublist = sub1 + sub_mid + sub2
            # if not_consecutive(sub1) and not_consecutive(sub2):
            print(sublist)
            if not_consecutive(sublist, key=lambda x:x[0]):
              l.append(sublist)
        # print(l)
        def summ(sublist):
            sum = 0
            for tup in sublist:
                sum += tup[1]
            return sum
        d = {summ(sub):sub for sub in l}
        # print(list(d))
        maximum = max([summ(sub) for sub in l], default=0)
        d[0] = []
        return d[maximum]
# if list is the max sum, return it

def find_max_sum4(lst):

    result = [sublist for sublist in (lst[x:x+size]  for size in range(1, len(lst)+1)
                                      for x in range(len(lst) - size + 1))]
    filtered = filter(lambda x: not_consecutive(x, key=lambda y: y[0]), result)
    return result

v = [ 1, 2, 3, 4, 5]
v1 = [1,3,4,6,8]
v2 = [1,2,3]
import itertools

# print(find_max([1,2,3]))
for sublist in find_max_sum4(list(enumerate(v2))):
    print(sublist)
# print(find_max_sum3(v))
# print(find_max_sum3(v1))

# print(find_max_sum([2, 5, 6, 5, 3]))