import math as math
import numpy as np
from linreg import LinReg
from matplotlib import pyplot as plt

iterations = 2048 * 256
print_rate = 8192
training_set_size = 2048
batch_size = 64

input_range = 16

training_set_inputs = np.random.uniform(high=input_range, size=(training_set_size, 1))
training_set_outputs = 2 * training_set_inputs[:, 0, np.newaxis] + 3

normalization_mean = 0
normalization_std_dev = 1
output_mean = 0
output_std_dev = 1


linReg = LinReg(1)

print('{:>10} | {:<24} | {:<24}'.format('iteration', 'cost', 'test input [[1, 2]]'))
s = ''
for i in range(10 + 24 + 24 + 2 * 3):
    s += '-'
print(s)

for i in range(iterations):
    k = math.floor(i % training_set_size / batch_size)
    batch = (training_set_inputs[k:k + batch_size] - normalization_mean) / normalization_std_dev
    batch_out = (training_set_outputs[k:k + batch_size] - output_mean) / output_std_dev
    linReg.train(batch, batch_out, 0.0001)
    if(i % print_rate == 0):
        print(
            '{:>10} | {:<24} | {:<24}'
            .format(i, linReg.cost(linReg.run(batch), batch_out), linReg.run(np.array([[2]]))[0, 0], flush=True))

print('final weight')
print(linReg._weight)

print('final test:')
print((linReg.run((np.array([[2]]) - normalization_mean) / normalization_std_dev) * output_std_dev + output_mean)[0, 0])
