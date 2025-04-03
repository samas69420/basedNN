import random
from net import Network
from algebruh import round_col
from algebruh import round_list
from algebruh import argmax_list


LEARNING_RATE = 0.001
NETWORK_LAYERS = [6,50,50,2]
N_ITERATIONS = 500_000
STATS_FREQ = 10_000


# define the activation function that will be used in the network 
# (and its derivative)
def lRelu(x): 
    return x if x>=0 else x*0.001
def lRelu_prime(x):
    return 1 if x>=0 else 0.001

#def relu(x):
#    return max(x,0)
#def relu_prime(x):
#    return 1 if x > 0 else 0

#import math
#def sigmoid(x):
#    return 1/(1+math.exp(-x))
#def sigmoid_prime(x):
#    return (1-sigmoid(x))*sigmoid(x)


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


def train_one_data_point(net, input_data, target):
    net.forward(input_data)
    net.backward(target)
    net.update_weights()


def test_n_samples(net, n):
    for _ in range(n):
        data = input_function()
        print("net output:",round_col(net.forward(data[0]),4),"- true output:", data[1],"- input:",round_list(data[0]))


def print_training_stats(net, i):
    hit_ratio = 0
    for _ in range(100):
        data = input_function()
        desired_out = data[1]
        out = net.forward(data[0]) 
        out_list = [e[0] for e in out]
        pred_argmax = argmax_list(out_list)
        desired_argmax = argmax_list(desired_out)
        if pred_argmax == desired_argmax:
            hit_ratio += 1
    print("iterations:",i,"accuracy:",hit_ratio/100)


if __name__ == "__main__":

    net = Network(NETWORK_LAYERS , lRelu, lRelu_prime, lr = LEARNING_RATE)

    # training loop
    for i in range(N_ITERATIONS):

        input_data, desired_output = input_function()
        train_one_data_point(net, input_data, desired_output)

        if i % STATS_FREQ == 0:
            print_training_stats(net, i)

    # test
    test_n_samples(net, 10)

