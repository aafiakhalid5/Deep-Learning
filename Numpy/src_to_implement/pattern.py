import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        if resolution % (2 * tile_size) != 0:
            raise ValueError(
                "Resolution should divide evenly into 2 * tile size")
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        base_tile = np.array([[0, 1], [1, 0]])  # classic checker tile
        num_tiles = self.resolution // (2 * self.tile_size)
        checker_pattern = np.tile(base_tile, (num_tiles, num_tiles))
        self.output = np.kron(checker_pattern, np.ones((self.tile_size, self.tile_size)))

        return self.output.copy()  # safer to return a copy

    def show(self):
        if self.output is not None:
            plt.imshow(self.output, cmap='gray')
            plt.title("Checker Pattern")
            plt.axis('off')
            plt.show()

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None  # again, for storing the image

    def draw(self):

        y = np.arange(self.resolution)
        x = np.arange(self.resolution)
        xx, yy = np.meshgrid(x, y)

        cx, cy = self.position
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        self.output = (dist2 <= self.radius ** 2)

        return self.output.copy()

    def show(self):
        if self.output is not None:
            plt.imshow(self.output, cmap='gray')
            plt.title("Circle Pattern")
            plt.axis('off')
            plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        # Making two gradients from 0 to 1
        horiz = np.linspace(0, 1, self.resolution)
        vert = np.linspace(0, 1, self.resolution)

        red, green = np.meshgrid(horiz, vert)
        blue = 1 - red

        self.output = np.stack((red, green, blue), axis=2)
        return self.output.copy()

    def show(self):
        if self.output is not None:
            plt.imshow(self.output)
            plt.axis('off')
            plt.show()
