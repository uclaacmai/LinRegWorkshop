import math as math
import numpy as np
from linreg import LinReg

iterations = 65536
print_rate = 8192
training_set_size = 128
batch_size = 4

training_set_inputs = np.random.uniform(high=16, size=(training_set_size, 2,))
training_set_outputs = training_set_inputs[:, 0, np.newaxis] + 2 * training_set_inputs[:, 1, np.newaxis] + 3

linReg = LinReg(2)

for i in range(iterations):
    k = math.floor(i % training_set_size / batch_size)
    batch = training_set_inputs[k:k + batch_size]
    batch_out = training_set_outputs[k:k + batch_size]
    linReg.train(batch, batch_out, 0.001)
    # linReg.train(training_set_inputs, training_set_outputs, 0.0001)
    if(i % print_rate == 0):
        print('iteration {:>8} | cost: {:>8}'.format(i, linReg.cost(linReg.run(batch), batch_out)), flush=True)

print('final weight')
print(linReg._weight)

print('final test')
print(linReg.run(np.array([[1, 2]])))
