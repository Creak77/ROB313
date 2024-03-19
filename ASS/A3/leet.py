def removeDuplicates(nums) -> int:
    d = set()
    res = []
    for i in range(len(nums)):
        if nums[i] not in d:
            d.add(nums[i])
            res.append(nums[i])
    while len(nums) > len(res):
        res.append(0)
    return res

print(removeDuplicates([1,2, 0]))