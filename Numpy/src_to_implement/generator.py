import os
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import random


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.file_path = file_path
        self.batch_size = batch_size
        self.image_size = image_size  # [H, W, C]
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.epoch = 0
        self.current_index = 0

        self.class_dict = {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
        }

        # Load labels
        with open(label_path, 'r') as f:
            self.labels_dict = json.load(f)

        self.filenames = list(self.labels_dict.keys())
        self.labels = [self.labels_dict[name] for name in self.filenames]
        self.num_samples = len(self.filenames)

        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        images = []
        labels = []

        for _ in range(self.batch_size):
            if self.current_index >= self.num_samples:
                self.epoch += 1
                self.current_index = 0
                if self.shuffle:
                    np.random.shuffle(self.indices)

            idx = self.indices[self.current_index]
            file_name = self.filenames[idx]
            label = self.labels_dict[file_name]

            # Load and resize
            img_path = os.path.join(self.file_path, f"{file_name}.npy")
            img = np.load(img_path)
            img = resize(img, self.image_size, preserve_range=True, anti_aliasing=True).astype(np.float32)

            # Apply augmentation
            if self.rotation or self.mirroring:
                img = self.augment(img)

            images.append(img)
            labels.append(int(label))  # Ensure it's an int

            self.current_index += 1

        return np.array(images), np.array(labels)

    def augment(self, img):
        # Random mirroring
        if self.mirroring and random.random() < 0.5:
            img = np.fliplr(img)

        # Random rotation
        if self.rotation:
            angle = random.choice([90, 180, 270])
            k = angle // 90
            img = np.rot90(img, k)

        return img

    def current_epoch(self):
        return self.epoch

    def class_name(self, label):
        return self.class_dict[label]

    def show(self):
        images, labels = self.next()
        cols = 5
        rows = int(np.ceil(self.batch_size / cols))
        plt.figure(figsize=(15, 3 * rows))
        for i in range(self.batch_size):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(images[i].astype(np.uint8))
            plt.title(self.class_name(labels[i]))
            plt.axis('off')
        plt.tight_layout()
        plt.show()
