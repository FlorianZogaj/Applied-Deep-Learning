import torch
import os
import torch
import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image

class Segmentor:

    def __init__(self, threshold=0.3, checkpoint="checkpoints/sam_vit_h_4b8939.pth"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Initialize the SAM model with the correct checkpoint and model type
        self.threshold = threshold
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        self.sam = sam.to(self.device)
        self.predictor = SamPredictor(sam)

    def convert_image(self, img_input):
        if isinstance(img_input, str):  # if the input is a file path
            image = cv2.imread(img_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(img_input, Image.Image):  # if input is a PIL image
            image = np.array(img_input)
        elif isinstance(img_input, np.ndarray):  # if input is a numpy array
            image = img_input
        else:
            raise TypeError("Unsupported image type")
        return image

    def __call__(self, img_input, bboxes):
        # Convert the image input to the required format for SAM
        image = self.convert_image(img_input)

        # Set the image for the SAM predictor
        self.predictor.set_image(image)
        # Convert list to a numpy array and reshape as expected by SAM
        # Obtain masks using the SAM predictor
        combined_mask = []
        for bbox in bboxes:
            mask, _, _ = self.predictor.predict(box=np.array([bbox]))
            combined_mask.append(np.any(mask, axis=0))
        # yield masks
        # combined_mask = np.any(combined_mask, axis=0)
        return combined_mask