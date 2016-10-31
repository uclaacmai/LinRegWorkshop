import numpy as np
from linreg import LinReg


training_set_inputs = np.random.uniform(high=1, size=(32, 2,))
training_set_outputs = 3*training_set_inputs[:,0,None] + 2*training_set_inputs[:,1,None] + 4

linReg = LinReg(2)

for i in range(256):
    batch = training_set_inputs[i%32,:,None]
    linReg.train(training_set_inputs, training_set_outputs, 0.01)

print('final weight')
print(linReg._weight)

print('final test')
print(linReg.run(np.array([[1, 2]])))
