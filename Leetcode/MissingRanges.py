"""
Given a sorted integer array where the range of elements are [lower, upper] inclusive, return its missing ranges.

For example, given [0, 1, 3, 50, 75], lower = 0 and upper = 99, return ["2", "4->49", "51->74", "76->99"].

https://leetcode.com/problems/missing-ranges/
"""


class MissingRanges:
    def __init__(self):
        self.missing_ranges = []

    def report_range(self, lower, upper):
        if lower == upper:
            self.missing_ranges.append(str(lower))
        else:
            self.missing_ranges.append('{}->{}'.format(lower, upper))

    def find_missing_ranges(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: List[str]
        """
        self.missing_ranges = []
        empty = not bool(nums)


        # add inclusive ranges into the list to process
        if empty or nums[0] != lower:
            nums.insert(0, lower-1)
        if empty or nums[-1] != upper:
            nums.append(upper+1)


        # Go through each number
        # If not consecutive, add the range that was missing
        for i in range(len(nums)):
            if i - 1 >= 0 and nums[i] != nums[i - 1] + 1:
                self.report_range(nums[i - 1] + 1, nums[i] - 1)

        return self.missing_ranges


from unittest import TestCase

test0 = []
test0a = []
test1 = [1, 2, 3] # 1, 3
test2 = [1, 4] # 1, 4
test3 = [0, 1, 3, 50, 75] # 0, 99
test4 = [1] # 1, 4
test5 = [-1] # -2, -1


class TestMissingRanges(TestCase):
    def test_find_missing_ranges(self):
        tester = MissingRanges()
        self.assertEqual(['1'], tester.find_missing_ranges(test0, 1, 1))
        self.assertEqual(['-1->4'], tester.find_missing_ranges(test0a, -1, 4))
        self.assertEqual([], tester.find_missing_ranges(test1, 1, 3))
        self.assertEqual(['2->3'], tester.find_missing_ranges(test2, 1, 4))
        self.assertEqual(["2", "4->49", "51->74", "76->99"], tester.find_missing_ranges(test3, 0, 99))
        self.assertEqual(['2->4'], tester.find_missing_ranges(test4, 1, 4))
        self.assertEqual(['-2'], tester.find_missing_ranges(test5, -2, -1))

