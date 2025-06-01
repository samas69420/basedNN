from algebruh import *

class ConvNet:

    def __init__(self, layers, learning_rate = 0.01):

        self.lr = learning_rate
        self.layers = layers

    def forward(self, input_3d_tensor):

        result = input_3d_tensor

        for layer in self.layers:
            result = layer.forward(result)        

        self.last_output = result

        return result

    def backward(self, desired_output):

        # cal dL/dO assuming MSE on the entire desired_output 3d-tensor
        d_O = []
        output_shape = get_tensor_shape(self.last_output)
        for s,matrix in enumerate(self.last_output):
            m = []
            for i,row in enumerate(matrix):
                r = []
                for j,element in enumerate(row):
                    r.append((2/(output_shape[0]*output_shape[1]*output_shape[2]))*(element-desired_output[s][i][j]))
                m.append(r)
            d_O.append(m)

        for layer in reversed(self.layers):
            d_O = layer.backward(d_O)
            

    def update_weights(self):

        for layer in self.layers:
            layer.update_weights(self.lr)

