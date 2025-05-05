from algebruh import *
import math


tanh = lambda x : math.tanh(x)
tanh_prime = lambda x : 1-(tanh(x)**2)
sigmoid = lambda x : 1/(1+math.exp(-x))
sigmoid_prime = lambda x : sigmoid(x)*(1-sigmoid(x))


class LSTM:


    def __init__(self, input_size, state_size, output_size, lr = 0.01):

        self.state_size = state_size
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.init_params()
        self.init_log_lists()


    def init_params(self):

        self.W_fx = random_matrix(self.state_size,self.input_size)
        self.W_fh = random_matrix(self.state_size,self.state_size)
        self.p_f = random_col(self.state_size)
        self.b_f = random_col(self.state_size)

        self.W_ix = random_matrix(self.state_size,self.input_size)
        self.W_ih = random_matrix(self.state_size,self.state_size)
        self.p_i = random_col(self.state_size)
        self.b_i = random_col(self.state_size)

        self.W_gx = random_matrix(self.state_size,self.input_size)
        self.W_gh = random_matrix(self.state_size,self.state_size)
        self.b_g = random_col(self.state_size)

        self.W_ox = random_matrix(self.state_size,self.input_size)
        self.W_oh = random_matrix(self.state_size,self.state_size)
        self.p_o = random_col(self.state_size)
        self.b_o = random_col(self.state_size)

        self.W_y = random_matrix(self.output_size,self.state_size)
        self.b_y = random_col(self.output_size)


    def init_log_lists(self):
        """
        list of variables to record in every timestep during forward cause
        they are needed for backward, the first element of each list
        represents the value at timestep "-1" (the initialization value)
        """

        self.h_log = [None]
        self.c_log = [None]
        self.y_log = [None]

        self.z_f_log = [None]
        self.z_i_log = [None]
        self.z_g_log = [None]
        self.z_o_log = [None]

        self.F_log = [None]
        self.I_log = [None]
        self.G_log = [None]
        self.O_log = [None]

        self.x_log = [None]


    def init_total_derivatives(self):
        """
        initialize all total derivatives to 0
        """

        self.d_W_fx = zero_matrix(len(self.W_fx),len(self.W_fx[0]))
        self.d_W_fh = zero_matrix(len(self.W_fh),len(self.W_fh[0]))
        self.d_p_f = zero_matrix(len(self.p_f),1)
        self.d_b_f = zero_matrix(len(self.b_f),1)

        self.d_W_ix = zero_matrix(len(self.W_fx),len(self.W_fx[0]))
        self.d_W_ih = zero_matrix(len(self.W_fh),len(self.W_fh[0]))
        self.d_p_i = zero_matrix(len(self.p_f),1)
        self.d_b_i = zero_matrix(len(self.b_f),1)

        self.d_W_gx = zero_matrix(len(self.W_fx),len(self.W_fx[0]))
        self.d_W_gh = zero_matrix(len(self.W_fh),len(self.W_fh[0]))
        self.d_b_g = zero_matrix(len(self.b_f),1)

        self.d_W_ox = zero_matrix(len(self.W_fx),len(self.W_fx[0]))
        self.d_W_oh = zero_matrix(len(self.W_fh),len(self.W_fh[0]))
        self.d_p_o = zero_matrix(len(self.p_f),1)
        self.d_b_o = zero_matrix(len(self.b_f),1)

        self.d_W_y = zero_matrix(len(self.W_y),len(self.W_y[0]))
        self.d_b_y = zero_matrix(len(self.b_y),1)


    def forward(self, x,h,c):

        # if this is the first forward of the sequence save init values for h/c
        if self.h_log[-1] == None and self.c_log[-1] == None:
            self.h_log[0] = h # h_bar
            self.c_log[0] = c # c_bar

        z_f = multi_mat_add( (mat_mul(self.W_fx,x), mat_mul(self.W_fh,h), element_wise_prod_col(self.p_f,c), self.b_f) )
        z_i = multi_mat_add( (mat_mul(self.W_ix,x), mat_mul(self.W_ih,h), element_wise_prod_col(self.p_i,c), self.b_i) )
        z_g = multi_mat_add( (mat_mul(self.W_gx,x), mat_mul(self.W_gh,h), self.b_g) )
        
        F = f_col(z_f, sigmoid)
        I = f_col(z_i, sigmoid)
        G = f_col(z_g, tanh)
        
        new_c = mat_add(element_wise_prod_col(F,c), element_wise_prod_col(I,G))
        
        z_o = multi_mat_add( (mat_mul(self.W_ox,x), mat_mul(self.W_oh,h), element_wise_prod_col(self.p_o,new_c), self.b_o) )
        O = f_col(z_o, sigmoid)
        
        new_h = element_wise_prod_col(f_col(new_c, tanh), O)
        y_hat = mat_add(mat_mul(self.W_y, new_h), self.b_y)

        # log variables needed for backprop, lists will be deleted in backward

        self.y_log.append(y_hat)
        self.h_log.append(new_h)
        self.c_log.append(new_c)

        self.z_f_log.append(z_f)
        self.z_i_log.append(z_i)
        self.z_g_log.append(z_g)
        self.z_o_log.append(z_o)

        self.F_log.append(F)
        self.I_log.append(I)
        self.G_log.append(G)
        self.O_log.append(O)

        self.x_log.append(x)

        return y_hat,new_h,new_c

    def backward(self, target_seq):
        """
        perform the bptt for the entire target sequence assuming the MSE loss
        """

        # initialize all the total gradient to 0
        self.init_total_derivatives()

        # set the derivatives for the last part of the sequence to 0
        # since for the last element F_t_plus_1 and other gates do not exist
        d_z_F_t_plus_1 = zero_matrix(self.state_size,1)
        d_z_I_t_plus_1 = zero_matrix(self.state_size,1)
        d_z_G_t_plus_1 = zero_matrix(self.state_size,1)
        d_z_O_t_plus_1 = zero_matrix(self.state_size,1)
        d_c_t_plus_1 = zero_matrix(self.state_size,1)
        self.F_log.append(zero_matrix(self.state_size,1))

        # bptt
        # t stops at 1 because the elements of index 0 in the log lists
        # corresponds to values at timestep "-1"
        for t in reversed(range(1,len(target_seq)+1)):

            # pick the values for the current timestep from log lists
        
            y_hat_t = self.y_log[t]
            h_t = self.h_log[t]
            c_t = self.c_log[t]
            h_t_minus_1 = self.h_log[t-1]
            c_t_minus_1 = self.c_log[t-1]

            z_f_t = self.z_f_log[t]
            z_i_t = self.z_i_log[t]
            z_g_t = self.z_g_log[t]
            z_o_t = self.z_o_log[t]

            F_t_plus_1 = self.F_log[t+1]
            I_t = self.I_log[t]
            G_t = self.G_log[t]
            O_t = self.O_log[t]

            x_t = self.x_log[t]

            # derivative of loss w.r.t. output at timestep t

            d_y_t = []
            for i in range(self.output_size):
                d_y_t.append([y_hat_t[i][0]-target_seq[t-1][i][0]])
            d_y_t = scal_mat_mult(2/self.output_size, d_y_t)

            # derivative of loss w.r.t. weights of final output layer

            d_W_y_t = outer_product(d_y_t,h_t)
            d_b_y_t = d_y_t
            
            # derivative of loss w.r.t. h_t

            d_h_t = multi_mat_add( (mat_mul(transpose(self.W_y), d_y_t),
                                    mat_mul(transpose(self.W_fh), d_z_F_t_plus_1),
                                    mat_mul(transpose(self.W_ih), d_z_I_t_plus_1),
                                    mat_mul(transpose(self.W_gh), d_z_G_t_plus_1),
                                    mat_mul(transpose(self.W_oh), d_z_O_t_plus_1)) )
            
            # derivative of loss w.r.t. weighted input to output gate

            d_z_O_t = element_wise_prod_col(element_wise_prod_col(d_h_t,f_col(c_t,tanh)),f_col(z_o_t,sigmoid_prime))
            
            # derivative of loss w.r.t. ct

            d_c_t = multi_mat_add( (element_wise_prod_col(d_z_O_t, self.p_o), 
                                    element_wise_prod_col(d_z_F_t_plus_1, self.p_f),
                                    element_wise_prod_col(d_z_I_t_plus_1, self.p_i),
                                    element_wise_prod_col(d_c_t_plus_1, F_t_plus_1),
                                    element_wise_prod_col(element_wise_prod_col(d_h_t, O_t), f_col(c_t, tanh_prime))) )
            
            # derivative of loss w.r.t. weighted input to forget/input/g gates 

            d_z_F_t = element_wise_prod_col(element_wise_prod_col(d_c_t, c_t_minus_1),f_col(z_f_t, sigmoid_prime))
            d_z_I_t = element_wise_prod_col(element_wise_prod_col(d_c_t, G_t),f_col(z_i_t, sigmoid_prime))
            d_z_G_t = element_wise_prod_col(element_wise_prod_col(d_c_t, I_t),f_col(z_g_t, tanh_prime))
            
            # params
            
            # derivative of loss w.r.t. weights of forget gate at timestep t

            d_W_fx_t = outer_product(d_z_F_t, x_t)
            d_W_fh_t = outer_product(d_z_F_t, h_t_minus_1)
            d_p_f_t = element_wise_prod_col(d_z_F_t, c_t_minus_1)
            d_b_f_t = d_z_F_t
            
            # derivative of loss w.r.t. weights of input gate at timestep t

            d_W_ix_t = outer_product(d_z_I_t, x_t)
            d_W_ih_t = outer_product(d_z_I_t, h_t_minus_1)
            d_p_i_t = element_wise_prod_col(d_z_I_t, c_t_minus_1)
            d_b_i_t = d_z_I_t
            
            # derivative of loss w.r.t. weights of g gate at timestep t

            d_W_gx_t = outer_product(d_z_G_t, x_t)
            d_W_gh_t = outer_product(d_z_G_t, h_t_minus_1)
            d_b_g_t = d_z_G_t
            
            # derivative of loss w.r.t. weights of output gate at timestep t

            d_W_ox_t = outer_product(d_z_O_t, x_t)
            d_W_oh_t = outer_product(d_z_O_t, h_t_minus_1)
            d_p_o_t = element_wise_prod_col(d_z_O_t, c_t)
            d_b_o_t = d_z_O_t

            # update the derivatives for next (previous) timestep

            d_z_F_t_plus_1 = d_z_F_t
            d_z_I_t_plus_1 = d_z_I_t
            d_z_G_t_plus_1 = d_z_G_t
            d_z_O_t_plus_1 = d_z_O_t
                          
            d_c_t_plus_1 = d_c_t

            # add contribution of timestep t to total derivatives

            self.d_W_fx = mat_add(self.d_W_fx, d_W_fx_t)
            self.d_W_fh = mat_add(self.d_W_fh, d_W_fh_t)
            self.d_p_f  = col_add(self.d_p_f, d_p_f_t)
            self.d_b_f  = col_add(self.d_b_f, d_b_f_t)

            self.d_W_ix = mat_add(self.d_W_ix, d_W_ix_t)
            self.d_W_ih = mat_add(self.d_W_ih, d_W_ih_t)
            self.d_p_i  = col_add(self.d_p_i, d_p_i_t)
            self.d_b_i  = col_add(self.d_b_i, d_b_i_t)

            self.d_W_gx = mat_add(self.d_W_gx, d_W_gx_t)
            self.d_W_gh = mat_add(self.d_W_gh, d_W_gh_t)
            self.d_b_g  = col_add(self.d_b_g, d_b_g_t)

            self.d_W_ox = mat_add(self.d_W_ox, d_W_ox_t)
            self.d_W_oh = mat_add(self.d_W_oh, d_W_oh_t)
            self.d_p_o  = col_add(self.d_p_o, d_p_o_t)
            self.d_b_o  = col_add(self.d_b_o, d_b_o_t)

            self.d_W_y  = mat_add(self.d_W_y, d_W_y_t)
            self.d_b_y  = col_add(self.d_b_y, d_b_y_t)

        # now that the gradients for the whole sequence have been computed the 
        # logs can be deleted
        self.init_log_lists()


    def update_weights(self):

        self.W_fx = mat_sub(self.W_fx, scal_mat_mult(self.lr, self.d_W_fx))
        self.W_fh = mat_sub(self.W_fh, scal_mat_mult(self.lr, self.d_W_fh))
        self.p_f  = col_sub(self.p_f , scal_col_mult(self.lr, self.d_p_f ))
        self.b_f  = col_sub(self.b_f , scal_col_mult(self.lr, self.d_b_f ))

        self.W_ix = mat_sub(self.W_ix, scal_mat_mult(self.lr, self.d_W_ix))
        self.W_ih = mat_sub(self.W_ih, scal_mat_mult(self.lr, self.d_W_ih))
        self.p_i  = col_sub(self.p_i , scal_col_mult(self.lr, self.d_p_i ))
        self.b_i  = col_sub(self.b_i , scal_col_mult(self.lr, self.d_b_i ))

        self.W_gx = mat_sub(self.W_gx, scal_mat_mult(self.lr, self.d_W_gx))
        self.W_gh = mat_sub(self.W_gh, scal_mat_mult(self.lr, self.d_W_gh))
        self.b_g  = col_sub(self.b_g , scal_col_mult(self.lr, self.d_b_g ))

        self.W_ox = mat_sub(self.W_ox, scal_mat_mult(self.lr, self.d_W_ox))
        self.W_oh = mat_sub(self.W_oh, scal_mat_mult(self.lr, self.d_W_oh))
        self.p_o  = col_sub(self.p_o , scal_col_mult(self.lr, self.d_p_o ))
        self.b_o  = col_sub(self.b_o , scal_col_mult(self.lr, self.d_b_o ))

        self.W_y  = mat_sub(self.W_y , scal_mat_mult(self.lr, self.d_W_y ))
        self.b_y  = col_sub(self.b_y , scal_col_mult(self.lr, self.d_b_y ))

