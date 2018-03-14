import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from random import  randint

class GridEncoder():

    def __init__(self):
        self.data = []
        self.maxlength = 10

    def generate_data_fake(self):
        vec = []
        for i in range(56859):
            tmp = []
            for k in range(29):
                tmp.append(randint(0, 9))
            vec.append(tmp)
        return np.asarray(vec)

    def generate_one_hot_vector(self, vec):
        oneHotEncoder = OneHotEncoder(self.maxlength).fit(vec)
        oneHotVector = oneHotEncoder.transform(vec)
        return oneHotVector