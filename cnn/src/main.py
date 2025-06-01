from conv2d import ConvLayer2d
from convdataset import get_random_image
from cnn import ConvNet
import random

random.seed(15)

# import math
#def relu(x):
#    return x if x >=0 else 0
#
#def relu_prime(x):
#    return 1 if x >=0 else 0
#
#def sigmoid(x):
#    return 1/(1+math.exp(-x))
#
#def sigmoid_prime(x):
#    return (1-sigmoid(x))*sigmoid(x)

def lRelu(x): 
    if x>=0:
        return x
    else:
        return x*0.01 

def lRelu_prime(x):
    if x>=0:
        return 1
    else:
        return 0.01 

if __name__ == "__main__":

    cnn = ConvNet([ConvLayer2d(1,15,(5,5), lRelu, lRelu_prime),
                   ConvLayer2d(15,10,(5,5), lRelu, lRelu_prime),
                   ConvLayer2d(10,5,(5,5), lRelu, lRelu_prime),
                   ConvLayer2d(5,1,(3,3), lRelu, lRelu_prime)],
                   learning_rate = 0.00001)

    for i in range(50_000):

        data, label = get_random_image()
        data = [data]
        label = [[[label]]]

        if i == 1000:
            cnn.lr = 0.0001

        out = cnn.forward(data)
        cnn.backward(label)
        cnn.update_weights()

        if i % 100 == 0:
            print(i, out)

    for i in range(10):
        data, label = get_random_image()
        data = [data]
        label = [[label]]
        
        out = cnn.forward(data)
        print(f"label: {label}, prediction:{out}")

# TODO
# add support to asymmetrical kernels' shape
# add auto padding when filter is bigger than input in forward
# refine the notes (add forward, full network arch etc)
# add support to variable stride
# add support to pooling layers (maxpool, meanpool)
# improve the logs during training and test
# add save/load weights feature
        
