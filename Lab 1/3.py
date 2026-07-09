l1=list(map(int,input("enter numbers of the first list: ").split(',')))
l2=list(map(int,input("enter numbers of the second list: ").split(',')))
x=len(l1)
y=len(l2)

large=0
small=0

if x>=y:
    large=l1
    small=l2
else:
    large=l2
    small=l1
count=0

for elem in large:
    if elem in small:
        count+=1
print("Number of common elements in both lists: ",count)