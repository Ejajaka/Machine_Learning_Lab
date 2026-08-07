import numpy as nump

def minkowski(x1, x2, p):
    # Generated with Claude
    x1 = nump.array(x1)
    x2 = nump.array(x2)
    return nump.sum(nump.abs(x1 - x2) ** p) ** (1 / p)

x1=nump.array([7, 3, 4])
x2=nump.array([17, 6, 9])
p=int(input("Enter the value of p: "))
if p==1 :
    print("The manhattan distance is: ", minkowski(x1, x2, p))
elif p==2 :
    print("The euclidean distance is: ", minkowski(x1, x2, p))
else:
    print("The minkowski distance is: ", minkowski(x1, x2, p))
