import csv
import math as math
import pickle as pickle
import numpy as np
from linreg import LinReg


with open('data/iris.data.txt') as f:
    dataset = np.array(list(csv.reader(f)))
    training_set_inputs = np.array(dataset[:, 0:3], dtype=np.float32)
    training_set_outputs = np.array(dataset[:, 3, np.newaxis], dtype=np.float32)

    print('training_set_inputs.shape')
    print(training_set_inputs.shape)
    print('training_set_outputs.shape')
    print(training_set_outputs.shape)

    rows, columns = training_set_inputs.shape

training_set_size = rows


#############
# Variables #
#############

iterations = 2048 * 256
print_rate = 8192
batch_size = 50
alpha = 0.0001  # default is 0.0001
reinit = True
save_file_path = 'data/iris.p'


##################
# Initialization #
##################

if reinit:
    linReg = LinReg(columns)
else:
    with open(save_file_path, 'rb') as p:
        weightinit = pickle.load(p)

        linReg = LinReg(columns, weightinit)


############
# Training #
############

print('{:>12} | {:<24}'.format('iteration', 'cost'))
s = ''
for i in range(12 + 24 + 3):
    s += '-'
print(s)

for i in range(iterations):
    k = math.floor(i % training_set_size / batch_size)
    batch = training_set_inputs[k:k + batch_size]
    batch_out = training_set_outputs[k:k + batch_size]
    linReg.train(batch, batch_out, alpha)

    if(i % print_rate == 0):
        print(
            '{:>12} | {:<24}'
            .format(i, linReg.cost(linReg.run(batch), batch_out)),
            flush=True)


########
# Save #
########

with open(save_file_path, 'wb') as p:
    pickle.dump(linReg._weight, p)

print('final weight')
print(linReg._weight)

test_range_min = 0
test_range_max = test_range_min + 64

print('final test')
test = linReg.run(training_set_inputs[test_range_min:test_range_max])
out = training_set_outputs[test_range_min:test_range_max]

print('{:<24} | {:<24} | {:<24}'.format('test', 'actual', 'difference'))
s = ''
for i in range(24 + 24 + 24 + 3 * 2):
    s += '-'
print(s)

for i, j in zip(test.flatten().tolist(), out.flatten().tolist()):
    print('{:<24} | {:<24} | {:<24}'.format(i, j, abs(i - j)))
