import gradio as gr
from PIL import Image
import numpy as np
from virtart.object_detector import ObjectDetector
from virtart.segment import Segmentor
from virtart.style_transfer import StyleTransfer
import warnings
import cv2
from tqdm import tqdm
warnings.filterwarnings('ignore')

def process_image(input_image, style_image):
    object_detector = ObjectDetector(device='gpu')
    sam = Segmentor()
    style_transfer = StyleTransfer()
    img_numpy = np.array(input_image)
    detections = object_detector(img_numpy)

    image = img_numpy[:, :, ::-1].copy()

    list = []
    for detection in detections:
        bbox = detection['bbox']
        label = detection['class']
        if detection['conf'] < 0.5:
            continue

        list.append(detection['class'])
        bbox = [int(box) for box in bbox]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.putText(image, label, (bbox[0], bbox[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    person_bboxes = [detection['bbox'] for detection in detections if detection['class'] == 'person']
    masks = sam(img_numpy, person_bboxes)

    return Image.fromarray(masks[0]), Image.fromarray(masks[1]), '\n'.join(list)


iface = gr.Interface(
    fn=process_image,
    inputs=["image", "image"],
    outputs=["image", "image", "text"])

iface.launch()
