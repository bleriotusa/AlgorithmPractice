import copy
import functools

import collections


class fiddle1:
    def __init__(self, name, value):
        self.name = name
        self.value = value


a = [fiddle1('john', 29), fiddle1('mary', 27)]

b = copy.copy(a)
c = copy.deepcopy(a)

a[0].value = 20

print('b', b[0].value)
print('c', c[0].value)


def bestDays(prices):
    min_price, max_profit = float('inf'), 0

    for price in prices:
        if price <= min_price:
            min_price = price
        else:
            profit = price - min_price
            max_profit = max(profit, max_profit)
    return max_profit


print(bestDays([5, 10, 0, 3, 4, 5]))

print(~0)

import string


def intToString(integer):
    answer = collections.deque()
    while True:
        answer.appendleft(string.digits[integer % 10])
        integer //= 10
        if integer == 0:
            break

    return "".join(answer)


def stringToInt(string):
    int_map = {'9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2, '1': 1, '0': 0}
    # answer = 0
    # for char in string:
    #     answer = int_map[char] + 10 * answer
    #     exp -= 1
    answer = functools.reduce(lambda answer, char: int_map[char] + 10 * answer, string, 0)
    return answer


print('int to string', intToString(123))
print('string to int', stringToInt("999"))


class node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


def merge_ll(l1, l2):
    # take care of null inputs here
    # ---

    l1_curr, l1_next = l1, l1.next
    l2_curr, l2_next = l2, l2.next

    head = l1 if l1.data <= l2.data else l2

    while l1_curr and l2_curr:
        if l1_curr.data <= l2_curr.data:
            l1_curr.next = l2_curr
            l1_curr = l1_next
            if l1_next:
                l1_next = l1_next.next
        else:
            l2_curr.next = l1_curr
            l2_curr = l2_next
            if l2_next:
                l2_next = l2_next.next

    return head


l1 = node(2, node(5, node(7, node(8))))
l2 = node(1, node(4, node(7, node(9, node(10, node(11))))))


def print_ll(ll):
    while ll:
        print(ll.data, '->', '', end='')
        ll = ll.next
    print('\n')


print_ll(merge_ll(l1, l2))
'''
2 -> 5 -> 7 -> 8
1 -> 6 -> 7 -> 9

1 -> 2 -> 3 -> 4 -> 7 -> 7 -> 8 -> 9

NOTE: None or 0 returns 0
'''


def football_combos(score):
    if score == 0:
        return 1
    elif score < 0:
        return 0

    def valid_score(sub_score):
        return sub_score >= 0 and (sub_score % 2 == 0 or sub_score % 3 == 0 or sub_score % 7 == 0)

    return (int(score - 2 > 0) + football_combos(score - 2)) if valid_score(score - 2) else 0 + \
                                                                                            (int(
                                                                                                score - 3 > 0) + football_combos(
                                                                                                score - 3)) if valid_score(
        score - 3) else 0 + \
                        (int(score - 7 > 0) + football_combos(score - 7)) if valid_score(score - 7) else 0


print(football_combos(2))
print(football_combos(3))
print(football_combos(7))
print(football_combos(12))

[5, 7, 2, 9, -2, 4, 10], 8
[5, 7, 2, 4]


def three_sum(l, parm):
    pass


def nearest_entries(array):
    pass


def dutch_partition(array):
    pass


def buy_sell_stock(prices):
    minimum = prices[0]
    max_profit = 0
    for price in prices[1:]:
        max_profit = max(price - minimum, max_profit)
        minimum = min(price, minimum)

    return max_profit

print('buy_sell', buy_sell_stock([310, 315, 275, 295, 260, 270, 290, 230, 255, 250]))

import functools
def string_to_int(num_string):
    '''
    '678' -> 678
    :param num_string: 
    :return: 
    '''
    number = 0
    for char in num_string:
        number *= 10
        number += string.digits.index(char)
    return number

def string_to_int_func(num_string):
    return functools.reduce(lambda answer, char_int: answer * 10 + string.digits.index(char_int), num_string, 0)

print('string_to_int of 678 ->', string_to_int('678'))
print('string_to_int_func of 678 ->', string_to_int_func('678'))

import collections
def int_to_string(num):
    '''
    678 -> '678'
    :param num: 
    :return: 
    '''

    answer_queue = collections.deque()
    while num:
        answer_queue.appendleft(string.digits[num % 10])
        num //= 10

    return ''.join(answer_queue)

print('int_to_string ->', int_to_string(678))


def merge_ll_2(ll1, ll2):
    if not ll1:
        return ll2

    head = current = node(-1)

    while ll1 and ll2:
        if ll1.data <= ll2.data:
            current.next, ll1 = ll1, ll1.next
        else:
            current.next, ll2 = ll2, ll2.next

        current = current.next

    if ll1 or ll2:
        current.next = ll1 or ll2

    return head.next

l1 = node(2, node(5, node(7, node(8))))
l2 = node(1, node(4, node(7, node(9, node(10, node(11))))))
print_ll(merge_ll_2(l1, l2))


class MaxStack:
    '''
    use two stacks. one stack for the data, one stack keeping track of the current max element
    '''
    def __init__(self):
        self.data = []
        self.max_stack = []

    def pop(self):
        result = not self.data or self.data.pop()
        if self.max_stack and result == self.max_stack[-1]:
            self.max_stack.pop()

        return result

    def push(self, element):
        if not self.max_stack or element >= self.max_stack[-1]:
            self.max_stack.append(element)

        self.data.append(element)

    def get_max(self):
        return self.max_stack[-1]


s = MaxStack()
s.push(6)
s.push(4)
s.push(8)
print('hello')
print(8, s.pop())
print(6, s.get_max())
print(4, s.pop())
print(6, s.pop())
print(None, s.pop())

def is_height_balanced(tree:node):

    def height_balanced_helper(tree:node):
        if not tree:
            return True, 0

        left, right = height_balanced_helper(tree.left), height_balanced_helper(tree.right)
        h_left, h_right = left[1], right[1]
        return left[0] and right[0] and bool(abs(left[1] - right[1] <= 1)), max(h_left, h_right) + 1

    return height_balanced_helper(tree)[0]

class BinaryTreeNode:

    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
# @exclude

    def __repr__(self):
        return '%s <- %s -> %s' % (self.left and self.left.data, self.data,
                                   self.right and self.right.data)

#  balanced binary tree test
#      3
#    2   5
#  1    4 6
print()
tree = BinaryTreeNode()
tree.left = BinaryTreeNode()
tree.left.left = BinaryTreeNode()
tree.right = BinaryTreeNode()
tree.right.left = BinaryTreeNode()
tree.right.right = BinaryTreeNode()
assert is_height_balanced(tree)
print(is_height_balanced(tree))
# Non-balanced binary tree test.
tree = BinaryTreeNode()
tree.left = BinaryTreeNode()
tree.left.left = BinaryTreeNode()
assert not is_height_balanced(tree)
print(is_height_balanced(tree))



def generate_permutations(array):
    '''
    [2, 3, 5, 7] has 4! permutations = 4 * 3 * 2 = 24
    :param array: 
    :return: 
    '''
    if not array:
        return [[]]

    # elif len(array) == 1:
    #     return [array]

    rest = generate_permutations(array[1:])
    print(rest)
    answer = []


    for permut in rest:
        if not permut:
            answer.append([array[0]])
        else:
            for i in range(len(permut)):
                answer.append(permut.insert(i, array[0]))

    return answer

print('permut ->', generate_permutations([2, 3, 5, 7]))


import random
random.choice
import math
def random_with_half(low, high):
    '''
    for an range of ints a-b, and a random num generator of 1 or 0, return a random number from the range
    :param array: 
    :return: Integer
    '''
    bin_digits = int(math.log(abs(high-low), 2)) + 1

    answer = ''
    while True:
        for i in range(bin_digits):
            answer += str(random.randrange(0,2))
        if int(answer, 2) + low <= high-low:
            break
        answer = ''
    return int(answer, 2) + low

print()
print('random', random_with_half(2, 10))

def evaluate_RPN(rpn_string):
    '''
    -34, 12 X 24 + --> -384
    (-2, ((6, 2, /), 2 +), X)
    :param rpn_string: 
    :return: 
    '''
    symbols = {'*', '/', '-', '+'}

    stack = []
    elements = [exp for exp in rpn_string.split(', ')]

    for item in elements:
        if item in symbols:
            b = stack.pop()
            a = stack.pop()
            stack.append(str(int(eval(a + item + b))))
        else:
            stack.append(item)

    return int(stack.pop())



def eval_RPN(rpn_exp):
    if ',' not in rpn_exp:
        if rpn_exp[0] == '-':
            return int(rpn_exp[1:]) * -1
        else:
            return int(rpn_exp)

    else:
        pass

print('eval_rpn:', evaluate_RPN('-2, 6, 2, /, 2, +, *'))

