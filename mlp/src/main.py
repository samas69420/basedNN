from net import Network
from dataset import input_function
from algebruh import round_list
from algebruh import argmax_list
from algebruh import col_to_list


# hyperparams
LEARNING_RATE = 0.001
NETWORK_LAYERS = [6,50,50,2]
N_ITERATIONS = 500_000

PRINT_STATS_FREQ = 10_000
SAVE_WEIGHTS_FREQ = 50_000

LOAD_EXISTING_WEIGHTS = True
EXISTING_WEIGHTS_FILE = "weights_6_50_50_2.w" 


def train_one_data_point(net, input_data, target):
    net.forward(input_data)
    net.backward(target)
    net.update_weights()


def test_n_samples(net, n, verbose = False):

    hit_ratio = 0

    for _ in range(n):

        input_data, desired_out = input_function()
        out = net.forward(input_data) 
        out_list = col_to_list(out)

        pred_argmax = argmax_list(out_list)
        desired_argmax = argmax_list(desired_out)

        if verbose:
            print("net output:",round_list(out_list,4),"- true output:", desired_out,"- input:",round_list(input_data))

        if pred_argmax == desired_argmax:
            hit_ratio += 1

    accuracy = hit_ratio/n 
    return accuracy


def print_training_stats(net,i):
    accuracy = test_n_samples(net, 100)
    print("iterations:",i,"accuracy:",accuracy)


if __name__ == "__main__":

    # define the activation function (and its derivative) that will be used 
    # in the network here
    def lRelu(x): 
        return x if x>=0 else x*0.001
    def lRelu_prime(x):
        return 1 if x>=0 else 0.001

    net = Network(NETWORK_LAYERS , lRelu, lRelu_prime, lr = LEARNING_RATE)

    if LOAD_EXISTING_WEIGHTS:
        print("loading weights from:",EXISTING_WEIGHTS_FILE)
        try:
            net.load_weights(EXISTING_WEIGHTS_FILE) 
        except FileNotFoundError:
            print(f"{EXISTING_WEIGHTS_FILE} not found")
            print("the weights file will be created automatically during train") 

    # training loop
    for i in range(1,N_ITERATIONS+1):

        input_data, desired_output = input_function()
        train_one_data_point(net, input_data, desired_output)

        if i % PRINT_STATS_FREQ == 0:
            print_training_stats(net, i)

        if i % SAVE_WEIGHTS_FREQ == 0:
            weights_file = net.save_weights()
            print("saved weights into:", weights_file)

    # test
    test_accuracy = test_n_samples(net, 10, verbose = True)
    print("final accuracy:", test_accuracy)

# TODO
# make a modular version of the mlp (linear layer class)
