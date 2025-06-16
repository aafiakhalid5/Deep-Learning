class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.weights = None  # Optional, for layers like FullyConnected
