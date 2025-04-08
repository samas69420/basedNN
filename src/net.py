from algebruh import * 
import struct

class Network:
    """
    class that implements a full-connected feed-forward deep neural network,
    it supports multi-layer architecture and a custom activation function but
    for now it assumes MSE error and the same activation function for all 
    the layers 
    """


    def __init__(self, neurons_list, activation_function, activation_function_derivative, lr=0.01):
        """
        takes a list of numbers representing the number of neurons for each 
        layer 

        (for example neuron_list = [10,500,100,100,2] means that there 
        will be 10 input neurons, a hidden layer of 500 neurons, two hidden 
        layers of 100 neurons and finally 2 output neurons)

        and the activation function (with its derivative)
        """

        self.neurons_list = neurons_list
        self.n_layers = len(neurons_list)
        self.f = activation_function
        self.f_prime = activation_function_derivative

        self.lr = lr

        # init weights and bias
        self.weights = []
        self.bias = []
        for i in range(self.n_layers-1):
            self.weights.append(random_matrix(neurons_list[i+1], neurons_list[i]))
            self.bias.append(random_matrix(neurons_list[i+1], 1))

        # these will be created and filled in forward and backward
        # they are included here only for readability
        self.weighted_input = None # z
        self.activations = None    # a
        self.gradients_w = None 
        self.gradients_b = None 

    
    def forward(self, x_list):
        """
        compute the forward pass but also record some data from each 
        layer that will be used later in the backprop
        """

        # clear the memory from data saved in previous forward (if present)
        self.weighted_input = [] # z
        self.activations = []    # a

        self.gradients_w = []
        self.gradients_b = []

        x_col = list_to_col(x_list)
        activations = x_col

        self.activations.append(activations)
        self.weighted_input.append(0) # 0 is appended here only to adjust size 

        for i in range(len(self.weights)):

            W = self.weights[i]
            b = self.bias[i]

            # compute the activations for the i-th layer
            weighted_input = col_add(mat_mul(W,activations), b)
            activations = f_col(weighted_input, self.f)

            # save the values computed for each layer
            self.weighted_input.append(weighted_input)
            self.activations.append(activations)

        # this will be needed in backprop
        self.last_input = x_col

        # quit if something explodes
        [quit() if repr(e[0])=="nan" else None for e in activations]

        return activations


    def backward(self, y_list):
        """
        compute the gradients for each layer in a recursive way starting from
        the last one (the output layer) assuming the MSE error function
        """

        gradients_weights = []
        gradients_bias = []

        x_col = self.last_input
        
        # loss = MSE => dl/do_i = 2/N (o_i-y_i)
        d = [[(self.activations[-1][i][0]-y_list[i])*(2/len(y_list))] for i in range(len(y_list))]

        # starting with the delta for the last layer
        activations_prime = f_col(self.weighted_input[-1], self.f_prime)
        delta = element_wise_prod_col(d, activations_prime)

        # compute gradients for hidden layers
        for i in reversed(range(self.n_layers-1)):
            
            # compute the gradients and save them in reverse order
            grad = outer_product(delta, self.activations[i])
            gradients_weights.append(grad)
            gradients_bias.append(delta)

            # compute delta for the previous layer but only if we are not at the input layer
            if i >0:
                WT_dot_delta = mat_mul(transpose(self.weights[i]),delta) 
                activations_prime = f_col(self.weighted_input[i],self.f_prime)
                delta = element_wise_prod_col(WT_dot_delta,activations_prime)

        gradients_weights.reverse()
        gradients_bias.reverse()

        self.gradients_w = gradients_weights
        self.gradients_b = gradients_bias

        # reset the lists for weighted input and activations to free some memory
        self.weighted_input = [] 
        self.activations = [] 


    def update_weights(self):
        """
        update all the learnable weights using the pure SGD algorithm: 
        W = W - lr * dL/dW
        b = b - lr * dL/db
        """

        for i in range(len(self.weights)):

            scaled_grad_w = scal_mat_mult(self.lr,self.gradients_w[i])
            new_W = mat_sub(self.weights[i], scaled_grad_w)
            self.weights[i] = new_W

            scaled_grad_b = scal_mat_mult(self.lr,self.gradients_b[i])
            new_b = col_sub(self.bias[i], scaled_grad_b)
            self.bias[i] = new_b

        # reset the lists of gradients
        self.gradients_w = []
        self.gradients_b = []

    
    def create_weights_filename(self, basename = "weights", extension = "w"):
        """
        append the current architecture to the basename and return the result
        for example if neurons_list = [6,50,50,2] and basename = weights:
        (weights,w) -> weights_6_50_50_2.w
        """ 

        neurons_list_str = ""
        for element in self.neurons_list:
            neurons_list_str += f"_{element}"

        formatted_filename = f"{basename}{neurons_list_str}.{extension}"
        return formatted_filename 

    
    def save_weights(self, filename = None):
        """
        save weights in a binary format (double precision) in a file
        if no filename is provided it will be generated automatically
        """

        if filename == None: 
            filename = self.create_weights_filename()
    
        with open(filename,"wb") as f:
            for i in range(len(self.neurons_list)-1):
                for i_ in range(self.neurons_list[i+1]):
                    for j_ in range(self.neurons_list[i]):
                        f.write(struct.pack('>d',self.weights[i][i_][j_]))

        return filename


    def load_weights(self, filename):
        """
        load weights from an existing file

        the architecture of the model when this function is used must be 
        exactly the same as the one used to generate the file (without 
        considering activation functions)
        """

        result = []

        with open(filename,"rb") as f:
            for i in range(len(self.neurons_list)-1):
                matrix = []
                for i_ in range(self.neurons_list[i+1]):
                    row = []
                    for j_ in range(self.neurons_list[i]):
                        next_8_bytes = f.read(8)
                        unpacked = struct.unpack('>d', next_8_bytes)[0]
                        row.append(unpacked)
                    matrix.append(row)
                result.append(matrix)
        
        self.weights = result


if __name__ == "__main__":

    # test save and load weights, activation functions are not needed 

    NETWORK_LAYERS = [2,3,2]
    net = Network(NETWORK_LAYERS , None, None)

    weights_file = net.save_weights("testweights.w")

    print("saved weights in", weights_file,"\n")
    print("first weight mat:")
    [print(row) for row in round_mat(net.weights[0])]

    print("\nresetting the net ...\n")
    net = Network(NETWORK_LAYERS , None, None)

    print("before loading\n")
    print("first weight mat:")
    [print(row) for row in round_mat(net.weights[0])]

    net.load_weights("testweights.w")

    print("\nafter loading\n")
    print("first weight mat:")
    [print(row) for row in round_mat(net.weights[0])]

