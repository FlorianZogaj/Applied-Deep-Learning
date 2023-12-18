# Virtual Art Gallery (Interactive Neural Style Transfer with Region-Specific Styling)
The goal of this project is to let users upload their photos and convert them into the styles of their choice. Neural Style transfer and Segmentation techniques are combined in this project to give the user more freedom in image styling.

![image](https://github.com/FlorianZogaj/Applied-Deep-Learning/assets/97000045/b9a953e1-3fd5-4082-b591-ae78811443f2)  
(source: https://forums.fast.ai/t/neural-style-transfer-using-s4tf/45128)


## 1. Type of Project
"Bring your own method" - Enhancing conventional neural style transfer with region-specific styling functionality. (Possibly combining with Segment Anything Model)  
Using pre-trained models that were trained on big datasets, the model can be altered to fit into the goal of this project. Further description of models and datasets that are in the scope of this project are discussed in the paper [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/pdf/2001.05566.pdf) [2]

## 2. Summary
An interactive application allowing users to apply the style from one image to specific regions of another image. While conventional neural style transfer alters the whole image, this project focuses on giving users the ability to stylize only certain regions. For example: I upload an image of myself and want to turn the background into a Starry Night Van Gogh Style. Depending on the complexity of the project, "Segment Anything Model" might also be implemented, allowing the user to specifically chose the areas to be styled. (https://segment-anything.com/)
For a real-time application, simpler architectures or those optimized for speed like DeepLabV3 or U-Net will also be considered.

![image](https://github.com/FlorianZogaj/Applied-Deep-Learning/assets/97000045/3bfa5015-1a76-4f71-b550-8e7955b6d629)  
Select dog and only style dog possible using SAM (source: https://segment-anything.com/)

## 3. Dataset description:
While the primary neural style transfer doesn't rely on extensive dataset training (using pretrained networks like VGG-19), for region-specific enhancements, we will use datasets which are typically used in segmentation tasks. A common dataset used to train segmentation models is for example the COCO dataset: https://cocodataset.org/#home

## 4. Work Breakdown Structure:
### 4.1 Dataset Collection & Project setup (7h) Deadline ~ 30.10
Find the necessary tools and models that can be used for this project.
Source and download relevant datasets.
Organize and preprocess images for testing.

### 4.2 Design and Build the Network (20h) Deadline ~ 16.11
Set up the basic neural style transfer using VGG-19 or a similar architecture.
Implement region-specific styling enhancements, integrating image segmentation.

### 4.3 Training and Fine-Tuning (17h) Deadline ~ 13.12
Using the pretrained models we fine-tune and iteratively test to ensure optimal region-specific stylization.

### 4.4 Application Development (6h) Deadline: 16.1
- Design an intuitive UI for image upload and region selection.
- Integrate the style transfer method into the application.
- Implement real-time previews.

### 4.5 Write the Final Report (10h) Deadline: 16.1
- Document the methodology, challenges faced, and solutions implemented.
- Highlight unique features and potential future enhancements.

### 4.6 Presentation Preparation (3h) Deadline: 16.1
- Design slides that showcase the project's features and outcomes.
- Prepare a demo, showing the application in action.

## 5. Related Literature
#### 1) Basic understanding of how style transfer works and how the network operates.
Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). **Image Style Transfer Using Convolutional Neural Networks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)**  
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

#### 2) Used to understand segmentation techniques and models. Furthermore the basic datasets that are used for training are explained
Minaee S., Boykov Y., Porikli F., Plaza A., Kehtarnavaz N., Terzopoulos D. (2020). **Image Segmentation Using Deep Learning: A Survey**
https://arxiv.org/pdf/2001.05566.pdf

#### 3) Get an understanding of what the SAM (Segment Anything Model) does
Kirillov A., Mintun E., Ravi N., Mao H., Rolland C., Gustafson L., Xiao T., Whitehead S., Berg A. C., Lo W., Dollar P., Girshick R. (2023) **Segment Anything**  
https://arxiv.org/pdf/2304.02643.pdf




# Further Todo's

- create tests
  - use amount of yolo bounding boxes vs real number (2 persons)
  - check iou of segment anything vs manual mask)
- yolo with test notebook for visualization
- sam with test notebook etc.
- style transfer
- improve style transfer with adjusting weight etc.
- plot graph of loss with varying influences
- fix resolution issue by dividing the image in multiple patches
- adjust readme with new samples and explanation

This repository borrows heavily from https://github.com/meituan/YOLOv6/tree/main ![yolov6]()


# Testing:
Have to use some kind of threshhold. If the detection at least detects everything "important" that is
on the image and in our classes of detected objects, we accept it.