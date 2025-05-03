import random
from algebruh import argmax_list

def input_function():
    """
    function that basically simulates an infinite dataset, every call
    returns x (the input) and y (the desired output for f(x))

    in this case the toy problem is to determine in which half of the input
    list the biggest element is so the x and y lists will be built in the
    following way:

    x = [x1, ... ,x6] every x is randomly generated in the interval (0,1)
    y = [1,0] if the biggest element is on the left otherwise [0,1]
    """

    x = [random.random() for i in range(6)]
    max_i = argmax_list(x)
    if max_i < 3:
        y = [1,0]
    else:
        y = [0,1]
    return x,y

