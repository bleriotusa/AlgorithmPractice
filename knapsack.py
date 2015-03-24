__author__ = 'Michael'

'''
Problem Statement

Given a list of n integers, A={a1,a2,…,an}, and another integer, k representing the expected sum. Select zero or more numbers from A such that the sum of these numbers is as near as possible, but not exceeding, to the expected sum (k).

Note

Each element of A can be selected multiple times.
If no element is selected then the sum is 0.
Input Format

The first line contains T the number of test cases.
Each test case comprises of two lines. First line contains two integers, n k, representing the length of list A and expected sum, respectively. Second line consists of n space separated integers, a1,a2,…,an, representing the elements of list A.

Constraints
1≤T≤10
1≤n≤2000
1≤k≤2000
1≤ai≤2000,where i∈[1,n]
Output Format

Output T lines, the answer for each test case.

Idea:
use the regular knapsack algorithm, then, for the case where a number is chosen to be in the set,
 add a for loop to iterate through all the multiples of the chosen number until the multiple is greater than the weight
'''

from decorators import *
import sys

class TailRecurseException(BaseException):
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs



def tail_call_optimized(g):
    """
    This function decorates a function with tail call
    optimization. It does this by throwing an exception
    if it is it's own grandparent, and catching such
    exceptions to fake the tail call optimization.

    This function fails if the decorated
    function recurses in a non-tail context.
    """

    def func(*args, **kwargs):
        f = sys._getframe()
        if f.f_back and f.f_back.f_back \
                and f.f_back.f_back.f_code == f.f_code:
            raise TailRecurseException(args, kwargs)
        else:
            while 1:
                try:
                    return g(*args, **kwargs)
                except TailRecurseException as e:
                    args = e.args
                    kwargs = e.kwargs
    func.__doc__ = g.__doc__
    return func

@Track_Calls
# @Memoize
# @Illustrate_Recursive
def knapsack(a: list, k: int):
    if not a or k <= 0:
        return 0
    else:
        # Two cases: if the last element is in the set, and if it's not.
        # find how close each case is to k, and return the sum of that set.
        case1 = knapsack(a[:-1], k - a[-1]) + a[-1] if a[-1] <= k else 0
        case2 = knapsack(a[:-1], k)

        diff1 = k - case1
        diff2 = k - case2

        if diff1 < 0 <= diff2:
            return case2
        elif diff2 < 0 <= diff1:
            return case1
        elif diff1 >= 0 and diff2 >= 0:
            return case1 if diff1 < diff2 else case2
        else:
            return 0


# @tail_call_optimized
# @Track_Calls
# @memoize
# @Illustrate_Recursive
def generic_knapsack(a: list, k: int):
    if not a or k <= 0:
        return 0
    all_multiple_values = []
    for i in range(0, int(k / a[-1]) + 1):
        total = generic_knapsack(a[:-1], k - i * a[-1]) + i * a[-1] if i * a[-1] <= k else 0
        diff = k - total

        # only consider answers less than or equal to k
        if diff >= 0:
            all_multiple_values.append((diff, total))

    # if no values in the list were < k, return 0
    if not all_multiple_values:
        return 0

    return min(all_multiple_values, key=lambda tup: tup[0])[1]


def iterative_knapsack(a: list, k: int):
    """
    unfinished
    """
    def knapsack_tail(a, k, acc):
        if not a:
            return acc
        else:
            case1 = knapsack_tail(a[:-1], k - a[-1], acc + a[-1]) if a[-1] <= k else 0
            case2 = knapsack_tail(a[:-1], k, acc)

            diff1 = k - case1
            diff2 = k - case2

            if diff1 < 0 <= diff2:
                return case2
            elif diff2 < 0 <= diff1:
                return case1
            elif diff1 >= 0 and diff2 >= 0:
                return case1 if diff1 < diff2 else case2
            else:
                return case2
    return knapsack_tail(a, k, 0)



if __name__ == '__main__':
    print("Initial Tests:")
    print("Knapsack")
    l = (5, 2, 10, 6)
    l2 = (5,)
    l3 = (5,)
    print(knapsack(l, 16))
    print("calls:", knapsack.calls)
    print(knapsack(l3, 15))
    print("calls:", knapsack.calls)

    print("\nIterative_Knapsack")
    print(iterative_knapsack(l, 16))
    # print("calls:", generic_knapsack.calls)
    print(iterative_knapsack(l3, 15))
    # print("calls:", generic_knapsack.calls)

    print("\nGeneric_Knapsack")
    print(generic_knapsack(l, 16))
    # print("calls:", generic_knapsack.calls)
    print(generic_knapsack(l3, 15))
    # print("calls:", generic_knapsack.calls)

    # Test Hackerrank test
    print("\nHackerrank test")
    sys.stdin = open('inputs/knapsack/in5.txt')
    sys.setrecursionlimit(2050)

    testcases = int(input())

    def get_line():
        return tuple(map(lambda num: int(num), input().strip().split(' ')))

    for i in range(0, testcases):
        n_k = get_line()
        k = n_k[1]
        nums = get_line()
        # print("Input Length:", len(nums))
        print(generic_knapsack(nums, k))