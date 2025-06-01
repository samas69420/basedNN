from algebruh import *


class ConvLayer2d:

    def __init__(self, in_channels, out_channels, filter_shape, 
                 activation_function = lambda x:x, 
                 activation_function_prime = lambda x:1):

        self.f = activation_function
        self.f_prime = activation_function_prime

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_shape = filter_shape

        self.kernels = []

        for out_channel in range(out_channels):
            kernel = []
            for in_channel in range(in_channels):
                kernel.append(random_matrix(filter_shape[0],filter_shape[1]))
            self.kernels.append(kernel)

        # there is a single bias (real number) for each channel of the output
        self.biases = [random.random()*2-1 for i in range(self.out_channels)]


    def forward(self, input_tensor):

        self.last_input = [e for e in input_tensor]

        input_shape = get_tensor_shape(input_tensor)
        output_shape_one_cha = (len(input_tensor[0])-self.filter_shape[0]+1,len(input_tensor[0][0])-self.filter_shape[1]+1)

        self.z = []
        result = []
        
        for k in range(self.out_channels):

            result_one_kernel = zero_matrix(output_shape_one_cha[0],output_shape_one_cha[1])

            for cha in range(self.in_channels):
                result_one_kernel = mat_add(result_one_kernel, conv2d(input_tensor[cha],self.kernels[k][cha]))

            result_one_kernel = mat_add(result_one_kernel, const_matrix(self.biases[k],output_shape_one_cha[0],output_shape_one_cha[1]))

            self.z.append(result_one_kernel)
            result.append(f_mat(result_one_kernel,self.f))

        return result


    def backward(self,d_O):
        """
        takes the derivative w.r.t. the output of the layer d_O and backpropagates it
        to the weights and the inputs
        """

        # gradients wrt weights

        self.gradients = []

        # add contribution of activation function to d_O
        new_d_O = []
        for i,element in enumerate(d_O):
            new_d_O.append(element_wise_prod_mat(element,f_mat(self.z[i], self.f_prime)))
        d_O = new_d_O

        for k in range(self.out_channels):
            gradient = []
            for channel in range(self.in_channels):
                gradient.append(conv2d(self.last_input[channel],d_O[k]))
            self.gradients.append(gradient)

        # gradients wrt bias
        
        self.gradients_bias = []
        for cha in range(self.out_channels):
            self.gradients_bias.append(sum_matrix(d_O[cha]))

        # gradients wrt input

        gradients_input_list = []

        for k in range(self.out_channels):
            gradient_one_kern = []
            for channel in range(self.in_channels):
                gradient_one_kern.append(conv2d(pad_matrix(d_O[k], size = len(self.kernels[0][0])-1), rotate_matrix_180(self.kernels[k][channel])))
            gradients_input_list.append(gradient_one_kern)
        
        gradient_input = [zero_matrix(len(self.last_input[0]),len(self.last_input[0][0])) for i in range(len(self.last_input))]
        for in_cha in range(self.in_channels):
            for out_cha in range(self.out_channels):
                gradient_input[in_cha] = mat_add(gradient_input[in_cha], gradients_input_list[out_cha][in_cha])

        return gradient_input
    

    def update_weights(self, lr): 

        for k in range(len(self.kernels)):
            for s in range(self.in_channels):
                self.kernels[k][s] = mat_sub(self.kernels[k][s], scal_mat_mult(lr,self.gradients[k][s]))

        for cha in range(len(self.biases)):
            self.biases[cha] -= lr*self.gradients_bias[cha]



