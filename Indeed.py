__author__ = 'Michael'
import math
def  getNumberOfPrimes( n):
    nums = 0
    for i in range(2, n+1):
        # print(i)
        prime = True
        for j in range(2, math.ceil(i**.5)+1):
            # print('j', j)
            if i % j == 0:
                prime = False
                break
        if prime:
            nums += 1
    return nums

print(getNumberOfPrimes(100))

def getNumberOfPrimes2(n):
    d = {i:False for i in range(2,n+1)}
    p = 2
    while p < n:
        mult = 2
        while mult*p < n:
            d[mult*p] = True
            mult += 1
        print('finished marking', p)
        for i in range(p+1, n):
            if d[i] is False:
                p = i
                break

    return [k for k, v in d.keys() if not v]


def getNumberOfPrimes3(n):
#     Let A be an array of Boolean values, indexed by integers 2 to n,
# initially all set to true.
    A = [True for i in range(2, n+1)]
    # print(A)
    for i in range(2, int(math.ceil(n**.5))):
        if A[i] is True:
            j = 0
            while i**2 + j*i < len(A):
                A[i**2 + j*i] = False
                j += 1
    B = [i for i in range(0, len(A)) if A[i]][2:]
    # print(B)
    return B
p = (getNumberOfPrimes3(100))
print(len(p), p)
# Output: all i such that A[i] is true.
print(len(getNumberOfPrimes3(1000000)))
# l = []
# l.extend([4]*100000000)
# l.extend([5]*100000000)
# l.extend([6]*100000000)

'''
def find_diff(l):
    first_two = l[1] - l[0]
    second_two = l[2] - l[1]
    return min(first_two, second_two)


# getting the number missing from arithmetic progression, python 2
n = input()
nums = raw_input()
ap = [int(i) for i in nums.split(' ')]
diff = find_diff(ap)
supposed = ap[0]

#print(ap)
#print(diff)

for i in range(0, n):
    current = ap[i]
    supposed = supposed + diff if i != 0 else supposed
    #print((current,supposed))
    if current != supposed:
        print(supposed)
        break

'''
'''
n = int(raw_input())
for i in range(0, n):
    a, b = raw_input().split()
    print int(a) + int(b)'''