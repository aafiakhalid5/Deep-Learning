import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self._prediction_tensor = None
        self._epsilon = np.finfo(float).eps  # Small constant to prevent log(0)

    def forward(self, prediction_tensor, label_tensor):
        self._prediction_tensor = prediction_tensor
        clipped_predictions = prediction_tensor + self._epsilon
        # Calculate loss only at positions where label == 1 (one-hot)
        loss = -np.sum(label_tensor * np.log(clipped_predictions))
        return loss

    def backward(self, label_tensor):
        # Gradient of Cross Entropy Loss: -y / (ŷ + ε)
        return -label_tensor / (self._prediction_tensor + self._epsilon)
