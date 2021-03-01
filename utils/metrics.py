import numpy as np


class AVGMetrics:
    def __init__(self, name):
        self.name = name
        self.values = []
        self.weights = []

    def update(self, value, weight=1):
        self.values.append(value)
        self.weights.append(weight)

    def __str__(self):
        return f"{self.name} is {self.__call__()}"

    def __call__(self):
        values = np.array(self.values)
        weights = np.array(self.weights) / sum(self.weights)
        avg = (values * weights).mean()
        return round(avg, 4)
