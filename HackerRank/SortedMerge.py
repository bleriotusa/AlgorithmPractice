__author__ = 'Michael'
"""
Sorted merge
Given 2 sorted arrays, merge them into one single sorted array and print its elements to standard output.
Expected complexity: O(N)
Example input:
a: 2 3 7 8 8
b: 7 8 13
Example output:
2 3 7 7 8 8 8 13
"""


def merge_arrays(a, b):
    # Write your code here
    # To print results to the standard output you can use print
    # Example: print "Hello world!"
    count_a = 0
    count_b = 0
    final_array = []
    unsorted = True
    while unsorted:
        if count_a == len(a) and count_b < len(b):
            final_array.append(b[count_b])
            print(b[count_b]),
            count_b += 1

        elif count_b == len(b) and count_a < len(a):
            final_array.append(a[count_a])
            print(a[count_a]),
            count_a += 1

        elif a[count_a] < b[count_b]:
            final_array.append(a[count_a])
            print(a[count_a]),1000000
            count_a += 1

        else:
            final_array.append(b[count_b])
            print(b[count_b]),
            count_b += 1

        if count_a == len(a) and count_b == len(b):
            unsorted = False

    return final_array

a = [2, 3, 7, 8, 8]
b = [7, 8, 13]

print(merge_arrays(a, b))