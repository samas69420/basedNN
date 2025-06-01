from algebruh import *
import random

vertical_line = zero_matrix(15,15)
for i in range(len(vertical_line)):
    for j in range(len(vertical_line[0])):
        if j in range(6,9):
            vertical_line[i][j] = 1

horizontal_line = zero_matrix(15,15)
for i in range(len(horizontal_line)):
    for j in range(len(horizontal_line[0])):
        if i in range(6,9):
            horizontal_line[i][j] = 1

X_shape = zero_matrix(15,15)
for i in range(len(X_shape)):
    for j in range(len(X_shape[0])):
        if (i==j or i==15-j-1) or \
           (i+1==j or i+1==15-j-1) or \
           (i-1==j or i-1==15-j-1):
            X_shape[i][j] = 1

diagonal_line_1 = zero_matrix(15,15)
for i in range(len(diagonal_line_1)):
    for j in range(len(diagonal_line_1[0])):
        if (i in range(j-1,j+2)):
            diagonal_line_1[i][j] = 1

diagonal_line_2 = zero_matrix(15,15)
for i in range(len(diagonal_line_2)):
    for j in range(len(diagonal_line_2[0])):
        if (i in range(14-j-1,14-j+2)):
            diagonal_line_2[i][j] = 1

circle = zero_matrix(15,15)
for i in range(len(circle)):
    for j in range(len(circle[0])):
        if (i-7)**2+(j-7)**2 < 56 and \
           (i-7)**2+(j-7)**2 > 25:
            circle[i][j] = 1

toy_dataset = [vertical_line,
               horizontal_line,
               X_shape,
               diagonal_line_1,
               diagonal_line_2,
               circle]

def get_random_image():
    index = random.randint(0,len(toy_dataset)-1)
    image = toy_dataset[index]
    return image, index

if __name__ == "__main__":
    for _ in range(6):
        data, label = get_random_image()
        print_mat(data)
        print("label:", label)
        print()
