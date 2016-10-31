import numpy as np
from linreg import LinReg

iterations = 524288
print_rate = 8192
training_set_size = 128

training_set_inputs = np.random.uniform(high=24, size=(training_set_size, 2,))
training_set_outputs = training_set_inputs[:,0,None] + 2*training_set_inputs[:,1,None] + 3

linReg = LinReg(2)

for i in range(iterations):
    batch = training_set_inputs[i%training_set_size,np.newaxis]
    batch_out = training_set_outputs[i%training_set_size,np.newaxis]
    linReg.train(batch, batch_out, 0.0001)
    # linReg.train(training_set_inputs, training_set_outputs, 0.0001)
    if(i % print_rate == 0):
        print('iteration {} cost: {}'.format(i, linReg.cost(linReg.run(batch), batch_out)), flush=True)

print('final weight')
print(linReg._weight)

print('final test')
print(linReg.run(np.array([[1, 2]])))
