"""
Given a binary tree, find the length of the longest consecutive sequence path.

The path refers to any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The longest consecutive path need to be from parent to child (cannot be the reverse).

For example,
   1
    \
     3
    / \
   2   4
        \
         5
Longest consecutive sequence path is 3-4-5, so return 3.
   2
    \
     3
    /
   2
  /
 1
Longest consecutive sequence path is 2-3,not3-2-1, so return 2.

https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/
"""
from Common.Tree import TreeNode


class BinaryTreeLongestConsecutiveSequence:
    """
    Dynamic Programming approach with recursion
    Base case, length 0 sequence
    Otherwise, solve left and right, and then return largest
    To solve:
    1. current node must be one less than sub node
    2. otherwise, reset the solution
    3. keep track of the longest sequence with global
    """

    def longestConsecutiveHelper(self, root):
        if not root:
            return 0

        left = self.longestConsecutiveHelper(root.left)
        right = self.longestConsecutiveHelper(root.right)

        if root.left and root.left.val == root.val + 1:
            left += 1
        else:
            left = 0
        if root.right and root.right.val == root.val + 1:
            right += 1
        else:
            right = 0

        longest = max(left, right, 1)
        self.maxx = max(longest, self.maxx)

        return max(left, right, 1)

    def longestConsecutive(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.maxx = 0
        self.longestConsecutiveHelper(root)
        return self.maxx


from unittest import TestCase

test0 = TreeNode(1)
test0.left = TreeNode(2)
test0.right = TreeNode(3)

test1 = TreeNode(1)
test1.right = TreeNode(2)
test1.right.right = TreeNode(5)
test1.right.right.right = TreeNode(1)
test1.right.right.right.right = TreeNode(2)
test1.right.right.right.right.right = TreeNode(3)

test2 = TreeNode(1)
test2.right = TreeNode(2)
test2.right.right = TreeNode(3)
test2.right.right.right = TreeNode(5)
test2.right.right.right.right = TreeNode(2)
test2.right.right.right.right.right = TreeNode(3)


class TestBinaryTreeLongestConsecutiveSequence(TestCase):
    def test_longestConsecutive(self):
        tester = BinaryTreeLongestConsecutiveSequence()
        self.assertEqual(2, tester.longestConsecutive(test0))
        self.assertEqual(3, tester.longestConsecutive(test1))
        self.assertEqual(3, tester.longestConsecutive(test2))
