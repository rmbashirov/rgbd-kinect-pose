import numpy as np
from copy import deepcopy


class DummyDict:
    def __init__(self, verbose=True):
        self.dummy_data = dict()
        self.verbose = verbose

    def set(self, k, v):
        if k not in self.dummy_data:
            self.dummy_data[k] = np.zeros_like(v)
        return v

    def get(self, k):
        if k not in self.dummy_data:
            if self.verbose:
                print(f'No dummy data for key {k}')
            return None
        else:
            return deepcopy(self.dummy_data[k])
