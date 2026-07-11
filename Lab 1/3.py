def count(l1, l2):
    if len(l1) >= len(l2):
        large = l1
        small = l2
    else:
        large = l2
        small = l1
    count = 0
    for elem in large:
        if elem in small:
            count += 1
    return count

l1 = list(map(int, input("Enter numbers of the first list: ").split(',')))
l2 = list(map(int, input("Enter numbers of the second list: ").split(',')))
count1 = count(l1, l2)
print("Number of common elements in both lists:", count1)