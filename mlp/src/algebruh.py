"""
in this file are implemented all the tensors operations but defined using 
only python lists

all "column" N-vector are treated as Nx1 matrices
all "row" N-vector are treated as 1xN matrices

functions labeled as "dummy" are just a wrapper for more general functions,
their purpose is only to help with readability such that the name of the
function itself tells us if we are dealing for example with column vectors rather
than row vectors even if in both cases the function for general matrices is used
to perform the computation
""" 

import random


def transpose(X):
    result = []
    for j in range(len(X[0])):
        row = []
        for i in range(len(X)):
            row.append(X[i][j])
        result.append(row)
    return result


def print_mat(X):
    for i in range(len(X)):
        for j in range(len(X[0])):
            print(X[i][j], end=" ")
        print()


def mat_mul(X,Y):
    result = []
    for i in range(len(X)):
        row = []
        for j_ in range(len(Y[0])):
            row_el = 0
            for j in range(len(X[0])):
                row_el += X[i][j]*Y[j][j_]
            row.append(row_el)
        result.append(row)
    return result


def element_wise_prod_list(x_list,y_list):
    result = []
    for i in range(len(x_list)):
        result.append(x_list[i]*y_list[i])
    return result


# dummy
def element_wise_prod_col(x_col,y_col):
    return element_wise_prod_mat(x_col, y_col)


def element_wise_prod_mat(X,Y):
    result = []
    for i in range(len(X)):
        row = []
        for j in range(len(X[0])):
            row.append(X[i][j]*Y[i][j])
        result.append(row)
    return result


def random_matrix(h,w, range_=[-1,1]):
    result = []
    for _ in range(h):
        row = [random.uniform(*range_) for i in range(w)]
        result.append(row)
    return result


# dummy
def random_col(size):
    return random_matrix(size,1)


def random_int_matrix(h,w, range_=[0,10]):
    result = []
    for _ in range(h):
        row = [random.randint(*range_) for i in range(w)]
        result.append(row)
    return result


def zero_matrix(h,w):
    result = []
    for _ in range(h):
        row = [0 for i in range(w)]
        result.append(row)
    return result


# dummy
def zero_col(size):
    return zero_matrix(size,1)


def f_col(x_col, f):
    result = [] 
    for element in x_col:
        result.append([f(element[0])])
    return result


def list_to_col(x_list):
    return [ [element] for element in x_list]


def col_to_list(x_col):
    return [e[0] for e in x_col]


def scal_mat_mult(k, X):
    result = []
    for i in range(len(X)):
        row = []
        for j in range(len(X[0])):
            row.append(k*X[i][j]) 
        result.append(row)
    return result


# dummy
def scal_col_mult(k, X):
    return scal_mat_mult(k,X)


def mat_sub(X,Y):
    result = []
    for i in range(len(X)):
        row = []
        for j in range(len(X[0])):
            row.append(X[i][j]-Y[i][j]) 
        result.append(row)
    return result


# dummy
def col_sub(x_col,y_col):
    return mat_sub(x_col,y_col) 
        

def mat_add(X,Y):
    result = []
    for i in range(len(X)):
        row = []
        for j in range(len(X[0])):
            row.append(X[i][j]+Y[i][j]) 
        result.append(row)
    return result


# dummy
def col_add(x_col,y_col):
    return mat_add(x_col,y_col) 


def multi_mat_add(mat_list):
    result = zero_matrix(len(mat_list[0]),len(mat_list[0][0]))
    for element in mat_list:
        result = mat_add(result, element)
    return result


# dummy 
def col_add(x_col,y_col):
    return mat_add(x_col,y_col) 


# not really dummy but almost
def outer_product(x_col, y_col):
    y_row = transpose(y_col)
    return mat_mul(x_col, y_row)


def round_mat(X,n=2):
    result = []
    for i in range(len(X)):
        row = []
        for j in range(len(X[0])):
            row.append(round(X[i][j],n)) 
        result.append(row)
    return result


def round_col(X,n=2):
    return round_mat(X,n)


def round_list(x_list,n=2):
    return [round(e,n) for e in x_list]


def argmax_list(x_list):
    max_el = max(x_list)
    for i,el in enumerate(x_list):
        if el == max_el:
            return i


if __name__ == "__main__":

    print("A")
    A_shape = [2,3]
    A = random_int_matrix(*A_shape)
    print_mat(A)
    print()

    print("B")
    B_shape = [3,2]
    B = random_int_matrix(*B_shape)
    print_mat(B)
    print()

    print("A*B")
    AB = mat_mul(A,B)
    print_mat(AB)
    print()

    print("C")
    C_shape = [2,2]
    C = random_matrix(*C_shape)
    print_mat(C)
    print()

    print("round C (2 decimal places)")
    Cr = round_mat(C)
    print_mat(Cr)
    print()

    print("AB * C element wise")
    D = element_wise_prod_mat(AB,C)
    print_mat(D)
    print()

    print("C1 (column vector)")
    C1_shape = [2,1]
    C1 = random_int_matrix(*C1_shape)
    print_mat(C1)
    print()

    print("R1 (row vector)")
    R1_shape = [1,2]
    R1 = random_int_matrix(*R1_shape)
    print_mat(R1)
    print()

    print("R1 * C1 (scalar product)")
    R1C1 = mat_mul(R1,C1)
    print_mat(R1C1)
    print()

    print("C1 * R1 (external product)")
    C1R1 = mat_mul(C1,R1)
    print_mat(C1R1)
    print()
    
    print("C2 (column vector)")
    C2_shape = C1_shape
    C2 = random_int_matrix(*C2_shape)
    print_mat(C2)
    print()
    print("C1 * C2 (element wise)")
    C1C2 = element_wise_prod_mat(C1,C2)
    C1C22 = element_wise_prod_col(C1,C2)
    print_mat(C1C2)
    print_mat(C1C22)

