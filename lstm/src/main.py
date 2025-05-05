from algebruh import random_col
from lstm import LSTM

INPUT_SIZE = 3
STATE_SIZE = 2
OUTPUT_SIZE = 3
LEARNING_RATE = 0.01

N_ITERATIONS = 100000

input_sequence = [[[0.1],
                   [0.2],
                   [0.3]],

                  [[0.4],
                   [0.5],
                   [0.6]],

                  [[0.7],
                   [0.8],
                   [0.9]]]

target_sequence = [[[2],
                    [6],
                    [9]],

                   [[1],
                    [4],
                    [20]],

                   [[3],
                    [3],
                    [3]]]

lstm = LSTM(INPUT_SIZE,STATE_SIZE,OUTPUT_SIZE,LEARNING_RATE)

h_bar = random_col(STATE_SIZE)
c_bar = random_col(STATE_SIZE)
h,c = h_bar,c_bar

print("before training:")
for i,element in enumerate(target_sequence):
    x = input_sequence[i]
    y,h,c = lstm.forward(x,h,c)
    print(y)

# call to backward because it also clears the log lists from previous forwards
lstm.backward(target_sequence)


print("training...")
for iteration in range(N_ITERATIONS):
    h,c = h_bar, c_bar
    for i,element in enumerate(target_sequence):
        x = input_sequence[i]
        y,h,c = lstm.forward(x,h,c)
    lstm.backward(target_sequence)
    lstm.update_weights()
    if iteration % (N_ITERATIONS/10) == 0:
        print(iteration)


print("after training:")
h,c = h_bar, c_bar
for i,element in enumerate(target_sequence):
    x = input_sequence[i]
    y,h,c = lstm.forward(x,h,c)
    print(y)


# NOTE 

# with these sequences of scalars the state size should be at least 3 in order to
# successfully learn the values, same thing for shorter sequences and probably
# with longer ones (not tested yet), however with sequences of 3-vectors like the
# ones used here STATE_SIZE = 2 seems to work 

# input_sequence = [[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]]]
# target_sequence = [[[6]],[[9]],[[4]],[[20]],[[13]],[[15]],[[7]],[[7]],[[7]],[[8]],[[8]],[[8]]]

