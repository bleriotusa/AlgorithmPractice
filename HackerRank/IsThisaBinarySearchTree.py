"""
For the purposes of this challenge, we define a binary search tree to be a binary tree with the following ordering properties:

The  value of every node in a node's left subtree is less than the data value of that node.
The  value of every node in a node's right subtree is greater than the data value of that node.
Given the root node of a binary tree, can you determine if it's also a binary search tree?

Complete the function in your editor below, which has  parameter: a pointer to the root of a binary tree. It must return a boolean denoting whether or not the binary tree is a binary search tree. You may have to write one or more helper functions to complete this challenge.

Note: We do not consider a binary tree to be a binary search tree if it contains duplicate values.

Input Format

You are not responsible for reading any input from stdin. Hidden code stubs will assemble a binary tree and pass its root node to your function as an argument.

Constraints

Output Format

You are not responsible for printing any output to stdout. Your function must return true if the tree is a binary search tree; otherwise, it must return false. Hidden code stubs will print this result as a Yes or No answer on a new line.
"""

""" Node is defined as
class node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
"""

from collections import defaultdict


class node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


d = defaultdict(int)


def check_binary_search_tree_(root):
    return is_binary_search_tree(root)


def memoize(f):
    cache = {}

    def wrapper(*args):
        if args in cache:
            return cache[args]
        answer = f(*args)
        cache[args] = answer
        return answer

    return wrapper


@memoize
def root_max(root):
    while (root.right):
        root = root.right

    return root.data


@memoize
def root_least(root):
    while (root.left):
        root = root.left

    return root.data


@memoize
def is_binary_search_tree(root):
    if not root:
        return True
    if d[root.data] > 0:
        return False
    else:
        d[root.data] += 1

    left_is_valid = ((not root.left) or (root.left.data < root.data and root_max(root.left) < root.data))
    right_is_valid = ((not root.right) or (root.right.data > root.data and root_least(root.right) > root.data))

    return left_is_valid and right_is_valid and is_binary_search_tree(root.left) and is_binary_search_tree(root.right)


class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return is_binary_search_tree(root)

# print(is_binary_search_tree(node(1)))
s = Solution()
print(s.isValidBST(node(1)))