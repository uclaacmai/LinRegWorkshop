import numpy as np


class LinReg:
    def __init__(self, size):
        '''
        # parameters
        size: int of weight length

        # properties
        weight: vertical vector of size + 1
        '''
        assert size > 0
        self.weight = np.random.normal(scale=0.5, size=(size + 1, 1,))
        print(self.weight)

    def reshape_inputs(self, inputs):
        return np.append(inputs, np.ones((inputs.shape[0], 1,)), axis=1)

    def run(self, inputs):
        '''Output of linear regression model
        # parameters
        inputs: numpy 2d array where rows are test case vectors
        '''
        return np.dot(self.reshape_inputs(inputs), self.weight)

    def train(self, inputs, outputs, alpha):
        '''trains weights
        # parameters
        inputs: numpy 2d array where rows are training case vectors
        outputs: numpy vertical vector of actual outputs
        alpha: training rate
        '''
        h = run(inputs)
        delta_weight = np.dot((h - outputs).T, self.reshape_inputs(inputs))*(-alpha/inputs.shape[0])
        self.weight += delta_weight
        return self.weight

    def cost(self, out_h, out_actual):
        '''computes squared cost
        # parameters
        out_h: numpy vertical vector of output from run
        out_actual: numpy vertical vector of actual output
        '''
        return np.sum((out_h - out_actual)**2)/2/out_h.shape[0]
