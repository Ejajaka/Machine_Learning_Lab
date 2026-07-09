r1 = int(input("Enter the number of rows for the first matrix: "))
c1 = int(input("Enter the number of columns for the first matrix: "))
r2 = int(input("Enter the number of rows for the second matrix: "))
c2 = int(input("Enter the number of columns for the second matrix: "))

if c1 != r2:
    print("Error: Matrix multiplication is not possible.")
    exit()

a = [[0 for j in range(c1)] for i in range(r1)]
b = [[0 for j in range(c2)] for i in range(r2)]
c = [[0 for j in range(c2)] for i in range(r1)]

for i in range(r1):
    for j in range(c1):
        print("Enter the element of first matrix at position [", i, "][", j, "]:")
        a[i][j] = int(input())
for i in range(r2):
    for j in range(c2):
        print("Enter the element of second matrix at position [", i, "][", j, "]:")
        b[i][j] = int(input())
        
for i in range(r1):
    for j in range(c2):
        for k in range(c1):
            c[i][j] += a[i][k] * b[k][j]

print("The resultant matrix is:")
for i in range(r1):
    for j in range(c2):
        print(c[i][j], end=" ")
    print()