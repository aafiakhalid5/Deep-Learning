from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(input_size + 1, output_size)
        self._optimizer = None
        self._gradient_weights = None
        self._input_tensor = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    @property
    def gradient_weights(self):
        return self._gradient_weights

    def forward(self, input_tensor):
        self._input_tensor = input_tensor
        bias = np.ones((input_tensor.shape[0], 1))
        extended_input = np.concatenate((input_tensor, bias), axis=1)
        self._extended_input = extended_input
        return extended_input @ self.weights

    def backward(self, error_tensor):
        self._gradient_weights = self._extended_input.T @ error_tensor
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        grad_input = error_tensor @ self.weights.T
        return grad_input[:, :-1]
