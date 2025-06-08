from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

def main():
    #Checker test
    checker = Checker(resolution=80, tile_size=10)
    checker.draw()
    checker.show()

    # Circle test
    circle = Circle(resolution=256, radius=60, position=(128, 80))
    circle.draw()
    circle.show()

    # Spectrum test
    spectrum = Spectrum(resolution=256)
    spectrum.draw()
    spectrum.show()

    #Generator
    file_path = "exercise_data"  # e.g., "./images/"
    label_path = "Labels.json"  # e.g., "./labels.json"
    batch_size = 10
    image_size = [32, 32, 3]

    generator = ImageGenerator(
        file_path, label_path, batch_size, image_size,
        rotation=True, mirroring=True, shuffle=True
    )

    generator.show()
    print("Current Epoch:", generator.current_epoch())

if __name__ == "__main__":
    main()
