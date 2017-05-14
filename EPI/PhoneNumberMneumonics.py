mapping = [0, 1, 'ABC', 'DEF', 'GHI', 'JKL', 'MNO', 'PQRS', 'TUV', 'WXYZ']


def find_combs(nums):
    """

    :rtype: List
    """
    if not nums:
        return []
    answer = []
    for char in mapping[int(nums[0])]:
        sub_sequences = [sequence for sequence in find_combs(nums[1:])]
        if sub_sequences:
            answer.extend([char + sequence for sequence in sub_sequences])
        else:
            answer.append(char)
    return answer

print(find_combs('2'))
print(find_combs('22'))
print(len(find_combs('2276696')), find_combs('2276696'))
