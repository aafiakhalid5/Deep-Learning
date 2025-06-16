import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self._input_tensor = None
        self._output_tensor = None

    def forward(self, input_tensor):
        # For numerical stability: subtract max along each row
        shifted_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_values = np.exp(shifted_input)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self._output_tensor = probabilities  # store for backward use
        return probabilities

    def backward(self, error_tensor):
        # Softmax gradient:
        # En-1 = ŷ * (En - sum_j(En_j * ŷ_j))
        batch_size, num_classes = error_tensor.shape
        dot = np.sum(error_tensor * self._output_tensor, axis=1, keepdims=True)
        return self._output_tensor * (error_tensor - dot)
