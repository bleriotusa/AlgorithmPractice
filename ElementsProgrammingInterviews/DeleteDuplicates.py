l = [2, 3, 5, 5, 7, 11, 11, 11, 13]

l1 = [2, 3, 5, 7, 11, 11, 11, 13]
def swap(array, i1, i2):
    temp = array[i1]



def delete_duplicates(array):
    if not array:
        return array

    empty_index = 1
    for i in range(1, len(array)):
        if empty_index != i:
            array[empty_index] = array[i]
        if array[i] == array[i-1]:
            empty_index = i
            while  i < len(array) and array[i] == array[i-1]:
                i += 1

    return list(set(array))

print(delete_duplicates(l))