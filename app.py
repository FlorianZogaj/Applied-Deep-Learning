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


def process_image(input_image, style):
    object_detector = ObjectDetector(device='gpu')
    sam = Segmentor()
    style_transfer = StyleTransfer()
    img_numpy = np.array(input_image)
    detections = object_detector(img_numpy)

    person_bboxes = [detection['bbox'] for detection in detections if detection['class'] == 'person']
    masks = sam(img_numpy, person_bboxes)

    result_img = np.array(input_image.copy())
    for mask, style in tqdm(zip(masks, [style, style])):
        styled_img, translation_mask, final_style_loss = style_transfer(img_numpy, mask, style)

        result_img[mask] = styled_img[translation_mask]

    height, width = result_img.shape[:2]

    crop_height = int(height * 0.1)
    crop_width = int(width * 0.1)

    top = crop_height
    bottom = height - crop_height
    left = crop_width
    right = width - crop_width

    cropped_img_array = result_img[top:bottom, left:right]

    return Image.fromarray(masks[0]), Image.fromarray(cropped_img_array)


iface = gr.Interface(
    fn=process_image,
    inputs=["image", gr.Image(type="pil")],
    outputs=["image", "image"])

iface.launch()
