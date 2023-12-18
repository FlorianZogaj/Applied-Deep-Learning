import torch
import os
import torch
import numpy as np
import cv2
import math
from segment_anything import SamPredictor, sam_model_registry
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer
from PIL import Image


class ObjectDetector:

    def __init__(self, checkpoint="yolov6n", device="cpu", half=False):
        self.checkpoint = checkpoint
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half = half and self.device != 'cpu'  # Only use half precision if the device is not CPU

        # Load model and other necessary components
        self._setup()

    def _setup(self):
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f"checkpoints/{self.checkpoint}.pt"
        if not os.path.exists(checkpoint_path):
            print(f"Downloading checkpoint {self.checkpoint}...")
            os.system(
                f"wget -c https://github.com/meituan/YOLOv6/releases/download/0.4.0/{self.checkpoint}.pt -O {checkpoint_path}")

        self.model = DetectBackend(checkpoint_path, device=self.device)
        self.stride = self.model.stride

        # Assuming class names are in a certain order
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ]

        if self.half:
            self.model.model.half()
        else:
            self.model.model.float()

        if self.device != 'cpu':
            img_size = [640, 480]  # default size, adjust if needed
            self.model(
                torch.zeros(1, 3, *img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

    def _prepare_image(self, img, img_size, stride):
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, str):
            img = cv2.imread(img)  # BGR format
            assert img is not None, f'Image Not Found {img}'

        # Letterbox image to desired size
        img = letterbox(img, img_size, stride=stride)[0]

        # Convert BGR to RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # Convert to Torch Tensor and normalize
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0  # normalize image
        if self.half:
            img = img.half()

        return img

    def __call__(self, img_input):
        img_size = 640  # default size, adjust if needed
        conf_thres = 0.25  # minimum confidence score a detection must have to be considered valid

        iou_thres = 0.45  # when 2 bounding boxes overlap. 2 boxes for the same object if IoU exceeded
        max_det = 1000
        agnostic_nms = False  # if we have overlapping objects of different classes

        # Adjust the image size if necessary
        img_size = self.check_img_size(img_size, s=self.stride)

        # Prepare the image
        img, img_src = Inferer.process_image(img_input, img_size, self.stride, False)

        if len(img.shape) == 3:
            img = img[None]  # add batch dimension

        # Inference
        img = img.to(self.device)
        with torch.no_grad():
            pred_results = self.model(img)
            det = non_max_suppression(pred_results, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)[0]

        # Process detections
        objects = []
        if len(det):
            # rescale back to original image
            det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape[:-1]).round()
            for *xyxy, conf, cls in reversed(det):
                cls = int(cls.item())
                object_class = self.class_names[cls]  # self.class_names.index('person')
                bbox = [x.item() for x in xyxy]  # [x1, y1, x2, y2] coordinates of top-left and bottom-right corners
                objects.append({
                    'class': object_class,
                    'bbox': bbox,
                    'conf': conf.item()
                })

                # if cls == self.class_names.index('person') and conf>0.5:
                #     # xywh = (Inferer.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     yield bbox, conf
        return objects

    def check_img_size(self, img_size, s=32, floor=0):
        # Make sure image size is a multiple of stride s in each dimension and return a new shape list of image
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size, list) else [new_size] * 2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor
        return math.ceil(x / divisor) * divisor
