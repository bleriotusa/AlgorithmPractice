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


from collections import defaultdict


def memoize(f):
    cache = {}

    def wrapper(*args):
        if args in cache:
            return cache[args]
        answer = f(*args)
        cache[args] = answer
        return answer

    return wrapper


class Solution(object):
    def __init__(self):
        self.d = defaultdict(int)

    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.is_binary_search_tree(root)

    @memoize
    def root_max(self, root):
        while (root.right):
            root = root.right

        return root.val

    @memoize
    def root_least(self, root):
        while (root.left):
            root = root.left

        return root.val

    @memoize
    def is_binary_search_tree(self, root):
        if not root:
            return True
        if self.d[root.val] > 0:
            return False
        else:
            self.d[root.val] += 1

        left_is_valid = ((not root.left) or (root.left.val < root.val and self.root_max(root.left) < root.val))
        right_is_valid = ((not root.right) or (root.right.val > root.val and self.root_least(root.right) > root.val))

        return left_is_valid and right_is_valid and self.is_binary_search_tree(root.left) and self.is_binary_search_tree(
            root.right)

# print(is_binary_search_tree(node(1)))

s = Solution()
print(s.isValidBST(node(1)))