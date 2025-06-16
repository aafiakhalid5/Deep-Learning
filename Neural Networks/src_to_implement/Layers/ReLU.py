import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        # No need to change trainable, it's already False in BaseLayer
        self._input_tensor = None

    def forward(self, input_tensor):
        self._input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        relu_derivative = self._input_tensor > 0  # mask where input > 0
        return error_tensor * relu_derivative
