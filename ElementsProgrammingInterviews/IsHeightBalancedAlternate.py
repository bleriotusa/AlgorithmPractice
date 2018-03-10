class BinaryTreeNode:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    # @exclude

    def __repr__(self):
        return '%s <- %s -> %s' % (self.left and self.left.data, self.data,
                                   self.right and self.right.data)


def is_height_balanced(root: BinaryTreeNode):
    def is_height_balanced_helper(root: BinaryTreeNode):
        if not root:
            return 0
        left = is_height_balanced_helper(root.left)
        right = is_height_balanced_helper(root.right)

        if left < 0 or right < 0 or abs(right - left) > 1:
            return -1

        return max(left + 1, right + 1)

    return is_height_balanced_helper(root) > 0


tree = BinaryTreeNode()
tree.left = BinaryTreeNode()
tree.left.left = BinaryTreeNode()
tree.right = BinaryTreeNode()
tree.right.left = BinaryTreeNode()
tree.right.right = BinaryTreeNode()
print(is_height_balanced(tree))
assert is_height_balanced(tree)
# Non-balanced binary tree test.
tree = BinaryTreeNode()
tree.left = BinaryTreeNode()
tree.left.left = BinaryTreeNode()
assert not is_height_balanced(tree)
print(is_height_balanced(tree))
