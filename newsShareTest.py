import csv
import math as math
import pickle as pickle
import numpy as np
from linreg import LinReg


with open('data/OnlineNewsPopularity.csv') as f:
    dataset = np.array(list(csv.reader(f)))
    training_set_inputs = np.array(dataset[1:, 2:60], dtype=np.float32)
    training_set_outputs = np.array(dataset[1:, 60, np.newaxis], dtype=np.float32)

    # print(training_set_inputs.shape) # (39645, 58)
    # print(training_set_outputs.shape) # (39645, 1)

    rows, columns = training_set_inputs.shape

training_set_size = rows


#############
# Variables #
#############

iterations = 2097152 * 1
print_rate = 8192
batch_size = 16
alpha = 0.001  # default is 0.0001
reinit = False

#################
# Normalization #
#################

normalization_mean = np.array(
    [
        10, 546, 0.54, 0.99, 0.68, 10, 3, 4, 1, 4,
        7, 0.05, 0.17, 0.15, 0.05, 0.18, 0.21, 26, 1153, 312,
        13612, 752324, 259281, 1117, 5657, 3235, 3998, 10329, 6401, 0.16,
        0.18, 0.18, 0.18, 0.14, 0.06, 0.06, 0.13, 0.18, 0.14, 0.21,
        0.22, 0.23, 0.44, 0.11, 0.03, 0.01, 0.68, 0.28, 0.35, 0.09,
        0.75, -0.25, -0.52, -0.1, 0.28, 0.07, 0.34, 0.15
    ], dtype=np.float32)
normalization_std_dev = np.array(
    [
        2, 471, 3.5, 5.2, 3.2, 11, 3.8, 8.3, 4.1, 0.84,
        1.9, 0.22, 0.38, 0.36, 0.23, 0.38, 0.4, 69, 3857, 620,
        57985, 214499, 135100, 1137, 6098, 1318, 19738, 41027, 24211, 0.37,
        0.38, 0.39, 0.38, 0.35, 0.24, 0.25, 0.33, 0.26, 0.21, 0.28,
        0.29, 0.28, 0.11, 0.09, 0.017, 0.01, 0.19, 0.15, 0.1, 0.07,
        0.24, 0.12, 0.29, 0.09, 0.32, 0.26, 0.18, 0.22
    ], dtype=np.float32)

output_mean = 3395
output_std_dev = 11626


##################
# Initialization #
##################

if reinit:
    linReg = LinReg(columns)
else:
    with open('data/newsshare.p', 'rb') as p:
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
    batch = (training_set_inputs[k:k + batch_size] - normalization_mean) / normalization_std_dev
    batch_out = (training_set_outputs[k:k + batch_size] - output_mean) / output_std_dev
    linReg.train(batch, batch_out, alpha)

    if(i % print_rate == 0):
        print(
            '{:>12} | {:<24}'
            .format(i, linReg.cost(linReg.run(batch), batch_out)),
            flush=True)


########
# Save #
########

with open('data/newsshare.p', 'wb') as p:
    pickle.dump(linReg._weight, p)

# print('final weight')
# print(linReg._weight)

test_range_min = 0
test_range_max = test_range_min + 64

print('final test')
test = (linReg.run((training_set_inputs[test_range_min:test_range_max] - normalization_mean) /
        normalization_std_dev) * output_std_dev) + output_mean
out = training_set_outputs[test_range_min:test_range_max]

print('{:>12} | {:>12} | {:>12}'.format('test', 'actual', 'difference'))
s = ''
for i in range(12 + 12 + 12 + 3 * 2):
    s += '-'
print(s)

for i, j in zip(test.flatten().tolist(), out.flatten().tolist()):
    print('{:>12} | {:>12} | {:>12}'.format(int(i), int(j), int(i - j)))
