import copy

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor  # Store for backward
        temp_output = input_tensor
        for layer in self.layers:
            temp_output = layer.forward(temp_output)
        return self.loss_layer.forward(temp_output, label_tensor)

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss_value = self.forward()
            self.loss.append(loss_value)
            self.backward()

    def test(self, input_tensor):
        temp_output = input_tensor
        for layer in self.layers:
            temp_output = layer.forward(temp_output)
        return temp_output  # Usually SoftMax output
