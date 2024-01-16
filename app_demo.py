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

object_detector = ObjectDetector(device='gpu')


def detect_objects(input_image):
    img_numpy = np.array(input_image)
    detections = object_detector(img_numpy)
    image = img_numpy[:, :, ::-1].copy()

    list = []
    for detection in detections:
        if detection['conf'] < 0.5:
            continue
        label = detection['class']
        bbox = detection['bbox']
        list.append(detection['class'])
        bbox = [int(box) for box in bbox]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(image, label, (bbox[0], bbox[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    image = image[:, :, ::-1]
    return Image.fromarray(image), list


def apply_style(input_image, style, selected_objects):
    sam = Segmentor()
    style_transfer = StyleTransfer()

    img_numpy = np.array(input_image)
    detections = object_detector(img_numpy)

    objects_bboxes = [detection['bbox'] for detection in detections if detection['conf'] > 0.5]

    masks = sam(img_numpy, objects_bboxes)
    result_img = img_numpy.copy()

    # for mask, style in tqdm(zip(masks, [style, style])):
    #   styled_img, translation_mask, final_style_loss = style_transfer(img_numpy, mask, style)
    #   result_img[mask] = styled_img[translation_mask]

    return Image.fromarray(masks[0])


def show_masks(input_image):
    sam = Segmentor()

    img_numpy = np.array(input_image)
    detections = object_detector(img_numpy)

    objects_bboxes = [detection['bbox'] for detection in detections if detection['conf'] > 0.5]

    masks = sam(img_numpy, objects_bboxes)

    return [Image.fromarray(masks[0]), Image.fromarray(masks[1]), Image.fromarray(masks[2])]


with gr.Blocks(theme=gr.themes.Soft()) as app:
    with gr.Row():
        with gr.Column():
            inp_image = gr.Image(type="numpy", label="Upload Content Image")
            style_image = gr.Image(type="pil", label="Upload Style Image")


        with gr.Column():
            detect_button = gr.Button("Detect Objects")
            masks_button = gr.Button("Generate masks")
            gallery = gr.Gallery(
                label="masks", show_label=False, elem_id="gallery"
                , object_fit="contain", height="auto")
            style_button = gr.Button("Apply Style Transfer")
            styled_image = gr.Image(label="Styled Image")

        with gr.Column():
            detected_image = gr.Image(label="Detected Objects")
            object_list = gr.Textbox()
            gr.Examples(
                examples=["imgs/neural_style_transfer_5_1.jpg", "imgs/wassily-kandinsky.jpg",
                          "imgs/great_wave.jpg", "imgs/scream.jpg"],
                inputs=style_image,
                outputs=style_image,
                fn=apply_style
            )

    detect_button.click(
        fn=detect_objects,
        inputs=inp_image,
        outputs=[detected_image, object_list]
    )
    masks_button.click(show_masks, inputs=inp_image, outputs=gallery)
    style_button.click(
        fn=apply_style,
        inputs=[inp_image, style_image, object_list],
        outputs=styled_image
    )

app.launch()
