"""
Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:

11110
11010
11000
00000
Answer: 1

Example 2:

11000
11000
00100
00011
Answer: 3

https://leetcode.com/problems/number-of-islands/
"""


class NumberOfIslands:
    def numIslands(self, grid):
        """
        for every coordinate,
            if island and not explored,
                add 1 to island count
                DFS in every direction
                    put explored coordinates in a tuple and into an explored set

        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid:
            return 0

        self.grid = grid
        self.explored = set()
        self.I = len(grid)
        self.J = len(grid[0])
        islands = 0
        for i in range(0, len(grid)):
            for j in range(0, len(grid[i])):
                coordinate = grid[i][j]
                # if island and not explored yet, mark as new island and explore it
                if coordinate and (i, j) not in self.explored:
                    islands += 1
                    self.DFS((i, j))

        return islands

    def DFS(self, tup):
        dfs_exp = set()
        dfs_to_exp = [tup]
        while dfs_to_exp:
            # explore next coord
            i, j = dfs_to_exp.pop()

            # if 0, we're in water
            if not self.grid[i][j]:
                continue

            # else, we're still on island so mark this coord as explored
            else:
                self.explored.add((i, j))
                # print('adding tup:', tup)
                dfs_exp.add((i, j))

            # add coords to search
            if i - 1 >= 0 and (i - 1, j) not in dfs_exp:
                dfs_to_exp.append((i - 1, j))
            if i + 1 < self.I and (i + 1, j) not in dfs_exp:
                dfs_to_exp.append((i + 1, j))
            if j - 1 >= 0 and (i, j - 1) not in dfs_exp:
                dfs_to_exp.append((i, j - 1))
            if j + 1 < self.J and (i, j + 1) not in dfs_exp:
                dfs_to_exp.append((i, j + 1))


from unittest import TestCase

test1 = [[1, 1, 1, 0], [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]
test2 = [[1,1,0,0], [1,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,1]]
test3 = [[1]]
test4 = [[1], [1]]
test5 = [[1], [0], [1]]
test6 = []

class TestNumberOfIslands(TestCase):
    def test_numIslands(self):
        test = NumberOfIslands()
        self.assertEqual(test.numIslands(test1), 1)
        self.assertEqual(test.numIslands(test2), 3)
        self.assertEqual(test.numIslands(test3), 1)
        self.assertEqual(test.numIslands(test4), 1)
        self.assertEqual(test.numIslands(test5), 2)
        self.assertEqual(test.numIslands(test6), 0)


    def test_numIslands2(self):
        test = NumberOfIslands()
        print(test.numIslands(test2))
