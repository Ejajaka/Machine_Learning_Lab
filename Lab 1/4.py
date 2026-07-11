def transpose(a, r1, c1):
    b = [[0 for j in range(c1)] for i in range(r1)]
    for i in range(r1):
        for j in range(c1):
            b[i][j] = a[j][i]
    return b

r1 = int(input("Enter the number of rows for the matrix: "))
c1 = int(input("Enter the number of columns for the matrix: "))
a = [[0 for j in range(c1)] for i in range(r1)]
for i in range(r1):
    for j in range(c1):
        print("Enter the element of first matrix at position [", i, "][", j, "]:")
        a[i][j] = int(input())
b = transpose(a, r1, c1)

print("The matrix is:")
for i in range(r1):
    for j in range(c1):
        print(a[i][j], end=" ")
    print()
print("The transpose of the matrix is:")
for i in range(r1):
    for j in range(c1):
        print(b[i][j], end=" ")
    print()