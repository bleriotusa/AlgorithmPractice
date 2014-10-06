def find_max_sum(v):
    # Write your code here
    # To print results to the standard output you can use print
    # Example: print "Hello world!"
    if len(v) == 1:
        return v[0]
    elif len(v) == 2:
        return max(v)
    elif len(v) == 3:
        return max(v)
    else:
        maximum = 0
        for i in range(0, len(v)):
            temp = find_max_sum(v[0:i]) + v[i] + find_max_sum(v[i+1:])
            if temp > maximum:
                maximum = temp
        return maximum

print(find_max_sum([2, 5, 6, 5, 3]))