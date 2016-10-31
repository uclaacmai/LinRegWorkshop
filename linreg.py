import numpy as np


class LinReg:
    def __init__(self, size):
        '''
        size: int of weight length
        '''
        assert size > 0
        self.weight = np.random.normal(size=(size,))
