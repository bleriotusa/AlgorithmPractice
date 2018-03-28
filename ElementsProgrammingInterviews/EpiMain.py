import copy
import functools
from heapq import heappush, heappop
import random
import string
from unittest import TestCase
import collections
import math
from Common.Tree import *

tester = TestCase()


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


def football_combos_nodp(score):
    twos = threes = sevens = 0
    curr_score = score

    if score % 2 == 0:
        twos += 1

    while curr_score > 0:
        if curr_score % 3 == 0:
            threes += 1
        curr_score -= 2

    curr_score = score
    while curr_score >= 7:
        curr_score -= 7
        sevens += football_combos_nodp(curr_score)

    return twos + threes + sevens


def football_combos_nodpe(score):
    def twos(score):
        return 1 if score % 2 == 0 else 0

    def threes(score):
        combos = 0
        while score > 0:
            if score % 3 == 0:
                combos += 1
            score -= 2
        return combos

    sevens = 0
    curr_score = score
    while curr_score >= 7:
        curr_score -= 7
        sevens += twos(score) + threes(score)

    return twos(score) + threes(score) + sevens


print('\nFootball Combos RIGHT')
print('For 2 points: ', football_combos_nodp(2))
print('For 3 points: ', football_combos_nodp(3))
print('For 7 points: ', football_combos_nodp(7))
print('For 9  points: ', football_combos_nodp(9))
print('For 12 points: ', football_combos_nodp(12))
print('For 14 points: ', football_combos_nodp(14))


def football_combos_dp(score):
    pass


[5, 7, 2, 9, -2, 4, 10], 8
[5, 7, 2, 4]


def three_sum(l, parm):
    pass


def nearest_entries(array):
    pass


def dutch_partition(array, index):
    array = array.copy()
    low = 0
    uncat = 0
    high = len(array) - 1
    pivot = array[index]

    while uncat <= high:
        if array[uncat] < pivot:
            array[low], array[uncat] = array[uncat], array[low]
            low += 1
            uncat += 1
        elif array[uncat] == pivot:
            uncat += 1
        else:
            array[high], array[uncat] = array[uncat], array[high]
            high -= 1

    return array


'''
 [0, 2, 1, 5, 0, 1]
 [0, 1, 1, 5, 0, 2]
 [0, 1, 1, 5, 0, 2]
 [0, 1, 1, 5, 0, 2]
 [0, 1, 1, 0, 5, 2]

 [3, 1, 4, 5, 5, 2, 9, 8]
 [3, 1, 4, 5, 2, 9, 8]
 [0, 2, 1, 0, 1, 5]

 '''


def check_dutch_partition(array, p):
    low = functools.reduce(lambda acc, e: acc + (1 if (e < p) else 0), array, 0)
    equal = functools.reduce(lambda acc, e: acc + (1 if (e == p) else 0), array, 0)
    for i in range(len(array)):
        if i < low:
            TestCase.assertLess(tester, array[i], p)
        elif equal and i < low + equal:
            TestCase.assertEqual(tester, array[i], p)
        else:
            TestCase.assertGreater(tester, array[i], p)


dutch_array = [0, 2, 1, 5, 0, 1]
dutch_index = 3
dutch_result = dutch_partition(dutch_array, dutch_index)

pivot = dutch_array[dutch_index]
print("\n\nDUTCH PARTITION\nINPUT: {}\nRESULT: {}".format(dutch_array, dutch_result))
check_dutch_partition(dutch_result, pivot)

dutch_index = 2
pivot = dutch_array[dutch_index]
dutch_result = dutch_partition(dutch_array, dutch_index)
print("\n\nDUTCH PARTITION\nINPUT: {}\nRESULT: {}\n\n".format(dutch_array, dutch_result))
check_dutch_partition(dutch_result, pivot)


def buy_sell_stock_once(prices):
    minimum = prices[0]
    max_profit = 0
    for price in prices[1:]:
        max_profit = max(price - minimum, max_profit)
        minimum = min(price, minimum)

    return max_profit


print('buy_sell', buy_sell_stock_once([310, 315, 275, 295, 260, 270, 290, 230, 255, 250]))


def buy_sell_stock_twice(prices):
    pass


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
        if self.data:
            result = self.data.pop()
        else:
            raise IndexError("pop from empty list")

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
try:
    print(None, s.pop())
except IndexError as err:
    print('Caught exception: ' + err.args[0])


def is_height_balanced(tree: node):
    def height_balanced_helper(tree: node):
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


# balanced binary tree test
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


def merge_sorted_files(files: [collections.Iterable]):
    """
    [3, 5, 7]
    [0, 6]
    [0, 6, 28]
    Start with heap with initial elements
    pop from heap into answer list, push from the list that it came from.
    :param files: list of iterables
    :return: list of sorted elements from files
    """
    heap = []
    result = []

    for i in range(len(files)):
        heappush(heap, (next(files[i]), i))

    while heap:
        smallest = heappop(heap)
        result.append(smallest[0])
        popped_index = smallest[1]
        try:
            next_item = (next(files[popped_index]), popped_index)
        except StopIteration:
            continue
        heappush(heap, next_item)

    return result


files_test = [iter([3, 5, 7]), iter([0, 6]), iter([0, 6, 28])]
files_test2 = [iter([3, 5, 7]), iter([0, 6]), iter([0, 6, 28])]
print("\n\nMerge Files: {}\nMerged Files: {}".format([list(file) for file in files_test],
                                                     merge_sorted_files(files_test2)))


def sorted_arrays_intersection(arr1, arr2):
    result = []
    i1 = i2 = 0
    while i1 < len(arr1) and i2 < len(arr2):
        if arr1[i1] == arr2[i2]:
            result.append(arr1[i1])
            i1, i2 = i1 + 1, i2 + 1
        elif arr1[i1] > arr2[i2]:
            i2 += 1
        else:
            i1 += 1
    return result


s_a1 = [0, 2, 3, 6]
s_a2 = [1, 2, 3, 8]
print('\n\nIntersection of Sorted Arrays: {}{}\nIntersection Array: {}'.format(s_a1, s_a2,
                                                                               sorted_arrays_intersection(s_a1, s_a2)))


def generate_permutations(array):
    """
    [2, 3, 5, 7] has 4! permutations = 4 * 3 * 2 = 24
    [[2, 3, 5, 7], [2, 3, 7, 5], [2, 5, 3, 7], [2, 5, 7, 3], [2, 7, 3, 5], [2, 7, 5, 3]]
    
    iterate through the array and swap the first element with the all the others so that all elements get a chance at 
    being first.
    
    while iterating, add the first element as the first element of every permutation returned by a recursive call
    to the rest of the list
    
    :param array: list of elements
    :return: list of lists (permutations) of the original list of elements
    """
    if not array:
        return []

    if len(array) == 1:
        return [array]

    permutations = []
    for i in range(len(array)):
        array[i], array[0] = array[0], array[i]
        permutations.extend([[array[0]] + permutation for permutation in generate_permutations(array[1:])])

    return permutations


print()
for permutation in enumerate(generate_permutations([2, 3, 5, 7])):
    print('permut {}-> {}'.format(permutation[0] + 1, permutation[1]))

print("distinct permutations:", len({tuple(permutation) for permutation in generate_permutations([2, 3, 5, 7])}))


def random_with_half(low, high):
    '''
    for an range of ints a-b, and a random num generator of 1 or 0, return a random number from the range
    :param array: 
    :return: Integer
    '''
    bin_digits = int(math.log(abs(high - low), 2)) + 1

    answer = ''
    while True:
        for i in range(bin_digits):
            answer += str(random.randrange(0, 2))
        if int(answer, 2) + low <= high - low:
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


print('eval_rpn:', evaluate_RPN('-2, 6, 2, /, 2, +, *'))


def inc_int_array(int_array):
    """
    [1, 2, 9] --> [1, 3, 0]
    [0] --> [1]
    [1, 9, 9, 9] --> [2, 0, 0, 0]
    [9] --> [1, 0]
    [2, 0] --> [2, 1]
    :param int_array: list of ints
    :return: list of ints
    """
    pass


import Common.Tree
from Common.Tree import BinarySearchTreeNode


def symmetric_bt(bt: BinarySearchTreeNode):
    """
    A tree is symmetric if the root's left and right are symmetric trees.
                100
             /          \
            50          50
        /       \     /     \  
      20        75  75      20
      /                        \
    10                            10  
    :param bt: 
    :return: 
    """

    def symmetric_bt_helper(bt_left, bt_right):
        """
        Base Cases:
        1. None, None --> OK
        2. True, True --> OK
        3. None, True or True, None --> FALSE
        
        Two BTs are symmetric if roots are equal, and:
        1. 1st Tree's Left is Symmetric to 2nd Tree's Right AND
        2. 1st Tree's Right is Symmetric to 2nd Tree's Left
        :param bt_left: 
        :param bt_right: 
        :return: bool
        """
        if not bt_left and not bt_right:
            return True

        elif (not bt_left and bt_right) or (bt_left and not bt_right):
            return False

        equal = bt_left.key = bt_right.key
        left_left_symm_right_right = symmetric_bt_helper(bt_left.left, bt_right.right)
        left_right_symm_right_left = symmetric_bt_helper(bt_left.right, bt_right.left)

        return equal and left_left_symm_right_right and left_right_symm_right_left

    return not bt or symmetric_bt_helper(bt.left, bt.right)


print()
# t = TreeNode(10, TreeNode(5, TreeNode(2), TreeNode(7)), TreeNode(15))
# print(Common.Tree.tree_to_list(t))
sym_tree = Common.Tree.list_to_tree(
    [100, [50, [20, [10, None, None], None], [75, None, None]], [50, [75, None, None], [20, None, [10, None, None]]]])
# Common.Tree.print_tree(sym_tree)
non_sym_tree = Common.Tree.list_to_tree([100, [50, [20, [10, None, [1, None, None]], None], [75, None, None]],
                                         [50, [75, None, None], [20, None, [10, None, None]]]])

print("sym tree is symmetric:", symmetric_bt(sym_tree))
print("non sym tree is not symmetric:", symmetric_bt(non_sym_tree))


def nearest_repeated_entries(words):
    """
    use a dict of the last known position of each word
    keep a min distance variable
    :param words: array representing paragraph
    :return: min distance between words
    """
    d = {}
    min_dist = float('inf')
    for i, word in enumerate(words):
        if word in d:
            min_dist = min(min_dist, i - d[word])

        d[word] = i

    return min_dist


sentence = "Hello there we are the best the best country so please be nice!".split(' ')
print()
print("Nearest repeated entries", nearest_repeated_entries(sentence))


def max_concurrent_events(events):
    """
    Given a list of events, we can check for the max events at the same time by
    1. Sorting the list of intervals by the starting interval 
    2. For each interval start/end point, check how many concurrent events are going on at that time
    
    Faster way is:
    1. Create list of endpoints from list of events (keeping info of start or end endpoint)
    2. Sort endpoints (with start points before endpoints if tie)
    3. Count up start points, subtract endpoints, keep track of max
    
    This uses an idea of augmenting your data into a more usable form
    :param events: 
    :return: integer max number of concurrent events
    """
    START = 0
    END = 1
    endpoints = []
    for event in events:
        endpoints.append((event[0], START))
        endpoints.append((event[1], END))

    max_count = curr_count = 0

    for endpoint, start_end in sorted(endpoints):
        if not start_end:  # this is a start
            curr_count += 1

        else:
            curr_count -= 1

        max_count = max(max_count, curr_count)

    return max_count


events = [(1, 4), (0, 5), (4, 8), (3, 4), (7, 10), (8, 11), (3, 6)]
print()
print("Max concurrent events:", max_concurrent_events(events))


def LCA_BST(bst: BinarySearchTreeNode, node1: BinarySearchTreeNode, node2: BinarySearchTreeNode):
    """
    Least common ancestor can be found by searching for both distinct nodes simultaneously, and remembering 
    the last node searched before the searches differ.
    
    candidate for coding
    :param bst:
    :param node1:
    :param node2:
    :return: node that is the LCA of node1 and node2
    """

    def bs_helper(bst: BinarySearchTreeNode):

        if node1.key < bst.key < node2.key or node1.key == bst.key or node2.key == bst.key:
            return bst

        elif node1.key < bst.key and bst.left:
            return bs_helper(bst.left)

        elif node1.key > bst.key and bst.right:
            return bs_helper(bst.right)

            # return bst

    LCA = bs_helper(bst)
    return LCA


bst_for_LCA = Common.Tree.list_to_tree(
    [100, [50, [25, None, [27, [26, None, None], [30, None, None]]], [75, None, None]],
     [150, [125, None, [128, None, None]], None]])
print()
print("LCA of BST should be 50:", LCA_BST(bst_for_LCA, BinarySearchTreeNode(30), BinarySearchTreeNode(75)).key)
print("LCA of BST should be 100:", LCA_BST(bst_for_LCA, BinarySearchTreeNode(26), BinarySearchTreeNode(128)).key)
print("LCA of BST should be 25:", LCA_BST(bst_for_LCA, BinarySearchTreeNode(25), BinarySearchTreeNode(30)).key)


def most_visited_page(file):
    """
    Use a combination of BST with keys of visit # / id tuples and a hashmap of ids to nodes of the bst
    :param file: 
    :return: 
    """
    pass


def memoize(f):
    cache, hits = {}, 0
    memoize.hits = 0

    def mem_func(*parms):
        if parms in cache:
            memoize.hits += 1
            return cache[parms]

        result = f(*parms)
        cache[parms] = result
        # print("hits", hits)
        return result

    return mem_func


@memoize
def traverse_2d_array(i, j):
    """
    Count number of ways to traverse from top left to bottom.
    BFS.
    Return 1 if bottom right, 0 if out of bounds, else the addition of all your subcalls. 
    
    Process:
    - Break up the problem space in to subproblems? couldn't.
    - How to solve this anyway? Optimize after.
    - How to get to the end? Try all ways starting from the top left. Seems like a graph search..
    - BST covers all your moves for each grid position. 
    - How to get answer? add up subcalls.. guess we have to return a value
    - If we return 1 at destination, seems like we can propogate that value all the way back up the call chain!
    - Out of bounds means that wasn't a path... = 0.
    
    wrong.
    step 1 was wrong, you can break the problem space into subproblems
    full solution = number of ways to get to square on left, + top 
    
    :param i: rows of array
    :param j: columns of array
    :return: 
    """

    if (i, j) == (0, 0):
        return 1

    if i < 0 or j < 0:
        return 0

    return traverse_2d_array(i, j - 1) + traverse_2d_array(i - 1, j)


import time

print()
print("paths to 2 by 2:", traverse_2d_array(1, 1))
print("paths to 3 by 3:", traverse_2d_array(2, 2))
print("paths to 5 by 5:", traverse_2d_array(4, 4))

start = time.perf_counter()
print("paths to 101 by 101:", traverse_2d_array(100, 100))
end = time.perf_counter()
print("Time:", end - start)
print("Cache Calls:", memoize.hits)


def knapsack_subset(items: list, capacity):
    """
    [(5g, $20), (7g, $50), (2g, $10), (3g, $25)], 13g
    :return: 
    """

    def subsets_valid_weight(items):
        if not items:
            return [[]]

        subsets_rest = subsets_valid_weight(items[1:])
        subsets_rest = list(
            filter(lambda subset: sum((item[0] for item in subset) if subset else [0]) <= capacity, subsets_rest))
        # print(subsets_rest)
        result = subsets_rest + [[items[0]] + subset for subset in subsets_rest]
        result = list(filter(lambda subset: sum((item[0] for item in subset) if subset else [0]) <= capacity, result))
        return [sorted(subset) for subset in result]

    max_subset = max(subsets_valid_weight(items), key=lambda subset: sum(item[1] for item in subset))
    return sum(item[1] for item in max_subset)


def subsets_valid_weight(items, capacity):
    if not items:
        return [[]]

    subsets_rest = subsets_valid_weight(items[1:], capacity)
    subsets_rest = list(
        filter(lambda subset: sum((item[0] for item in subset) if subset else [0]) <= capacity, subsets_rest))
    # print(subsets_rest)
    result = subsets_rest + [[items[0]] + subset for subset in subsets_rest]
    result = list(filter(lambda subset: sum((item[0] for item in subset) if subset else [0]) <= capacity, result))
    return [sorted(subset) for subset in result]


print()
items = [(5, 20), (7, 50), (2, 10), (3, 25)]
print('subsets_valid_weight', subsets_valid_weight(items, 13))
print('knapsack:', knapsack_subset(items, 13))


# @memoize
def knapsack(items, capacity):
    if not items or capacity < 0:
        return 0

    elif len(items) == 1:
        return items[0][1] if items[0][0] <= capacity else 0

    knapsack_rest_nofirst = knapsack(items[1:], capacity)
    knapsack_rest_first = knapsack(items[1:], capacity - items[0][0])
    return max(knapsack_rest_nofirst, knapsack_rest_first + items[0][1])


@memoize
def knapsack_set(items, capacity):
    """
    get answers of knapsack with the first item of list and without the first item
    if the first item's weight is <= capacity, get the max of the two, otherwise, just return the answer for the rest
    :param items: 
    :param capacity: 
    :return: 
    """
    if not items or capacity < 0:
        return ()

    elif len(items) == 1:
        return items if items[0][0] <= capacity else ()

    knapsack_rest_nofirst = knapsack_set(items[1:], capacity)
    knapsack_rest_first = knapsack_set(items[1:], capacity - items[0][0])
    return max(knapsack_rest_nofirst, (items[0],) + knapsack_rest_first,
               key=lambda knapsack: sum(item[1] for item in knapsack)) if items[0][
                                                                              0] <= capacity else knapsack_rest_nofirst


print()
items = ((5, 20), (7, 50), (2, 10), (3, 25))
items2 = (
    (20, 65), (8, 35), (60, 245), (55, 195), (70, 150), (85, 275), (25, 155), (30, 120), (65, 320), (75, 75), (10, 40),
    (95, 200), (50, 100), (40, 220), (10, 99))
items3 = (
    (20, 65), (8, 35), (60, 245), (55, 195), (70, 150), (85, 275), (25, 155), (30, 120), (65, 320), (75, 75), (10, 40),
    (95, 200), (50, 100), (40, 220), (10, 99), (22, 65), (82, 35), (61, 245), (555, 195), (72, 150), (86, 275),
    (27, 155), (38, 120), (653, 320), (72, 75), (12, 40),
    (90, 200), (53, 100), (42, 220), (11, 99))
print('knapsackk:', knapsack_set(items, 13))
k2 = knapsack_set(items2, 130)
print('knapsackk:', knapsack_set(items2, 130), sum(item[0] for item in k2), sum(item[1] for item in k2))
print('knapsackk:', knapsack_set(items3, 10000), sum(item[0] for item in k2), sum(item[1] for item in k2))
print("Cache Calls:", memoize.hits)


def majority_element(strings):
    """
    [a, a, b, b, a, a, b, a]
    [s, a, s, a, a, a, c, a]
    [a, b, a, b, a, a, b, a]
    :return: 
    """
    if not strings:
        return ""

    candidate, candidate_count = strings[0], 0

    for word in strings:
        if word == candidate:
            candidate_count += 1
        else:
            if candidate_count == 0:
                candidate = word
            else:
                candidate_count -= 1

    return candidate


print()
print("Majority element is a:", majority_element(['a', 'a', 'b', 'b', 'a', 'a', 'b', 'a']))
print("Majority element is a:", majority_element(['s', 'a', 's', 'b', 'a', 'a', 'c', 'a']))
print("Majority element is a:", majority_element(['a', 'b', 'a', 'b', 'a', 'a', 'b', 'a']))


def kth_node(bst: SizeBinarySearchTreeNode, k):
    """
                    100
                /           \
            50                  150
        /       \           /         \
    25          75                      175
            /                              \
        60                                     185
                                              /
                                        180
    
        #  size field
    #      6
    #    2   3
    #  1    1 1
    #
    #  data field
    #      3
    #    2   5
    #  1    4 6
    :param bst: 
    :return: 
    """

    while bst:
        size_left = bst.left.size if bst.left else 0

        if size_left + 1 == k:
            return bst

        elif k <= size_left:
            bst = bst.left

        else:
            k -= size_left + 1
            bst = bst.right

    return None


root = SizeBinarySearchTreeNode(3)
root.size = 6
root.left = SizeBinarySearchTreeNode(2)
root.left.size = 2
root.left.left = SizeBinarySearchTreeNode(1)
root.left.left.size = 1
root.right = SizeBinarySearchTreeNode(5)
root.right.size = 3
root.right.left = SizeBinarySearchTreeNode(4)
root.right.left.size = 1
root.right.right = SizeBinarySearchTreeNode(6)
root.right.right.size = 1
print()
assert None is kth_node(root, 0)
# should output 1
assert kth_node(root, 1).key == 1
print("kth node:", (kth_node(root, 1)).key)
# should output 2
assert kth_node(root, 2).key == 2
print("kth node:", (kth_node(root, 2)).key)
# should output 3
assert kth_node(root, 3).key == 3
print("kth node:", (kth_node(root, 3)).key)
# should output 4
assert kth_node(root, 4).key == 4
print("kth node:", (kth_node(root, 4)).key)
# should output 5
assert kth_node(root, 5).key == 5
print("kth node:", (kth_node(root, 5)).key)
# should output 6
assert kth_node(root, 6).key == 6
print("kth node:", (kth_node(root, 6)).key)
assert None is kth_node(root, 7)

for i in range(10):
    pass


def well_formed():
    """
    stack with lefts, right side symbols pop top if it's the corresponding left symbol.
    :return: 
    """
    pass


test = [0]
print(test[1:])


def n_queens(n):
    """
    Enumerate placements that use distinct rows. Mostly uses recursion for backtracking.
        Basically only try one in a row at a time, that isn't in: a used column, a diagonal.
        Diagonals mathematically is slope = 1 or -1. Slope = x2 - x1 / y2 - y1.. so you can just
        check if abs(x2 - x1) = abs(y2 - y1)
    :return:
    """
    result, col_placement = [], [-1] * n

    def solve_n_queens(row):
        if row == n:
            result.append(list(col_placement))
        else:
            # try every column
            for col in range(n):
                # if none in same column or diagonal so far, add a queen to this column
                # then do the same for the rest of the rows
                if all(abs(col - c) not in (0, row - i) for i, c in enumerate(col_placement[:row])):
                    col_placement[row] = col
                    solve_n_queens(row + 1)

    solve_n_queens(0)
    return result


print("\n\nn queens:")
print(n_queens(2))
print(n_queens(4))
TestCase.assertEqual(tester, n_queens(2), [])
TestCase.assertEqual(tester, n_queens(4), [[1, 3, 0, 2], [2, 0, 3, 1]])


def next_key_of_bst(bst: BinarySearchTreeNode, key: int, candidate=None):
    if not bst:
        return candidate

    if key < bst.key:
        candidate = bst
        return next_key_of_bst(bst.left, key, candidate)

    else:
        return next_key_of_bst(bst.right, key, candidate)


print()
print()
print('Figure 14.1 Tree: ')
figure_14_1_tree = [19, [7, [3, [2], [5]], [11, [], [17, [13], []]]],
                    [43, [23, [], [37, [29, [], [31]], [41, [], []]]], [47, [], [53, [], []]]]]
next_key_bst_test = list_to_tree(figure_14_1_tree)
next_key_bst_test.print_tree()

print('Next key after 23: ', next_key_of_bst(next_key_bst_test, 23, None).key)
print('Next key after 13: ', next_key_of_bst(next_key_bst_test, 13, None).key)
TestCase.assertEqual(tester, 29, next_key_of_bst(next_key_bst_test, 23, None).key)
TestCase.assertEqual(tester, 17, next_key_of_bst(next_key_bst_test, 13, None).key)


def search_for_sequence(table: [list], pattern: tuple):
    cache = {}
    search_for_sequence.hits = 0
    """
    Check for the start of the pattern match in all nodes in table
    :param table: 
    :param pattern: 
    :return: 
    """

    def search_for_sequence_helper(i, j, offset):
        """
        Check bounds
        Base case is if the pattern is one element and the current node is that element
        General case is Current Element == Next Pattern Element + Recurse on all positions around
        :param i:
        :param j:
        :param pattern[offset:]:
        :return:
        """
        # check cache
        if (i, j, offset) in cache.keys():
            search_for_sequence.hits += 1
            return cache[(i, j, offset)]

        # check boundaries
        if i < 0 or j < 0 or i >= len(table) or j >= len(table[i]):
            cache[(i, j, offset)] = False
            return False

        # base case
        if len(pattern[offset:]) == 1 and table[i][j] == pattern[offset:][0]:
            cache[(i, j, offset)] = True
            return True

        # general case
        match = table[i][j] == pattern[offset] and (
                search_for_sequence_helper(i + 1, j, offset+1) or search_for_sequence_helper(i, j + 1, offset+1)
                or search_for_sequence_helper(i - 1, j, offset+1) or search_for_sequence_helper(i, j - 1, offset+1))
        cache[(i, j, offset)] = match
        return match

    for i in range(len(table)):
        for j in range(len(table[0])):
            if search_for_sequence_helper(i, j, 0):
                return True

    return False


print()
search_sequence_table = [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
search_sequence_pattern = (1, 3, 4, 6)
print("Search for {} in {}: {}".format(search_sequence_pattern, search_sequence_table,
                                       search_for_sequence(search_sequence_table, search_sequence_pattern)))
print("Cache hits:", search_for_sequence.hits)
search_sequence_table = [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
search_sequence_pattern = (1, 2, 3, 4)
print("Search for {} in {}: {}".format(search_sequence_pattern, search_sequence_table,
                                       search_for_sequence(search_sequence_table, search_sequence_pattern)))
print("Cache hits:", search_for_sequence.hits)
search_sequence_table = [[1, 2]]
search_sequence_pattern = (1, 2)
print("Search for {} in {}: {}".format(search_sequence_pattern, search_sequence_table,
                                       search_for_sequence(search_sequence_table, search_sequence_pattern)))
print("Cache hits:", search_for_sequence.hits)
