class BinarySearchTreeNode(object):
    def __init__(self, x, left=None, right=None):
        self.key = x
        self.left = left
        self.right = right

    def print_tree(self, indent_char='.', indent_delta=2):
        atree = self

        def print_tree_1(indent, atree):
            if atree == None:
                return None
            else:
                print_tree_1(indent + indent_delta, atree.right)
                print(indent * indent_char + str(atree.key))
                print_tree_1(indent + indent_delta, atree.left)

        print_tree_1(0, atree)


class SizeBinarySearchTreeNode(BinarySearchTreeNode):
    def __init__(self, x, left=None, right=None, size=0):
        super().__init__(x, left, right)
        self.size = size


def list_to_tree(alist):
    if not alist:
        return None
    elif len(alist) == 1: # avoid index errors if we represent a childless node like [3]
        return BinarySearchTreeNode(alist[0])
    else:
        return BinarySearchTreeNode(alist[0], list_to_tree(alist[1]), list_to_tree(alist[2]))


def tree_to_list(atree):
    if atree == None:
        return None
    else:
        return [atree.key, tree_to_list(atree.left), tree_to_list(atree.right)]


def print_tree(atree, indent_char='.', indent_delta=2):
    def print_tree_1(indent, atree):
        if atree == None:
            return None
        else:
            print_tree_1(indent + indent_delta, atree.right)
            print(indent * indent_char + str(atree.key))
            print_tree_1(indent + indent_delta, atree.left)

    print_tree_1(0, atree)
