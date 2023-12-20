import unittest
from PIL import Image
import numpy as np
from virtart.object_detector import ObjectDetector
import warnings


# Testing for some selected images if the number of detected objects is right

class ObjectDetectionTest(unittest.TestCase):
    warnings.filterwarnings("ignore")

    def test_image1(self):
        object_detector = ObjectDetector(device='gpu')
        img = Image.open('../imgs/image1.jpg')
        img_numpy = np.array(img)

        detections = object_detector(img_numpy)

        persons = sum(1 for d in detections if d['class'] == 'person' and d['conf'] >= 0.70)
        cars = sum(1 for d in detections if d['class'] == 'car' and d['conf'] >= 0.3)

        self.assertEqual(persons, 2, "Number of persons detected does not match")
        self.assertEqual(cars, 1, "Number of cars detected does not match")

    def test_image2(self):
        object_detector = ObjectDetector(device='gpu')
        img = Image.open('../imgs/dog_bike_car.jpg')
        img_numpy = np.array(img)

        detections = object_detector(img_numpy)

        # Count detections only for interesting objects in this image
        dogs = sum(1 for d in detections if d['class'] == 'dog' and d['conf'] >= 0.50)
        cars = sum(1 for d in detections if d['class'] == 'car' and d['conf'] >= 0.5)
        bicycles = sum(1 for d in detections if d['class'] == 'bicycle' and d['conf'] >= 0.5)

        self.assertEqual(dogs, 1, "Number of dogs detected does not match")
        self.assertEqual(cars, 1, "Number of cars detected does not match")
        self.assertEqual(bicycles, 1, "Number of bicycles detected does not match")

    def test_image3(self):
        object_detector = ObjectDetector(device='gpu')
        img = Image.open('../imgs/Phuket.jpg')
        img_numpy = np.array(img)

        detections = object_detector(img_numpy)

        # Count detections only for interesting objects in this image
        persons = sum(1 for d in detections if d['class'] == 'person' and d['conf'] >= 0.30)

        self.assertEqual(persons, 16, "Number of persons detected does not match")

    def test_image4(self):
        object_detector = ObjectDetector(device='gpu')
        img = Image.open('../imgs/cat_standing.jpg')
        img_numpy = np.array(img)

        # Run object detection
        detections = object_detector(img_numpy)

        # Count detections only for interesting objects in this image
        cats = sum(1 for d in detections if d['class'] == 'cat' and d['conf'] >= 0.50)

        self.assertEqual(cats, 1, "Number of cats detected does not match")

    def test_image5(self):
        object_detector = ObjectDetector(device='gpu')
        img = Image.open('../imgs/CR7_Messi.jpg')
        img_numpy = np.array(img)

        detections = object_detector(img_numpy)

        # Count detections only for interesting objects in this image
        persons = sum(1 for d in detections if d['class'] == 'person' and d['conf'] >= 0.50)
        chairs = sum(1 for d in detections if d['class'] == 'chair' and d['conf'] >= 0.50)

        self.assertEqual(persons, 2, "Number of persons detected does not match")
        self.assertEqual(chairs, 1, "Number of chairs detected does not match")


if __name__ == '__main__':
    unittest.main()
