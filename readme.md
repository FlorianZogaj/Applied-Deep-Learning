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


# Assignment 2 explanation:

This repository borrows from https://github.com/meituan/YOLOv6/tree/main (yolov6 directory)



# Object Detection:
Regarding the object detection I have created some tests of my images to ensure the more interesting
objects (not considering too small objects or objects too far in background) are correctly detected.
As the plan is to make an application where a user can upload a picture and let his objects be detected
and then transfer a certain style on those, it might be less important to correctly classify the label
of the object but rather classify the objects at all with the right mask.


# Error Metric:
I manually created masks of objects in my images and compared them to the masks created by SAM.
In this case the manually created masks are the "ground truth" that is being compared to by IoU.
If an object is detected and I want to transfer a style to that object, I would want the mask to
be as close as possible to the real object. How good the style is transferred may be more subjective
and can be explored via user studies. However, by looking at the loss of style, I was still able to compare 
how the changes in style weight affected the image.
The goal was to achieve IoU scores of over 75% to ensure that good results are possible when transferring styles.
In testing_masks.ipynb the IoU values of 2 images are shown. Here I detected the 2 persons on the image and created their masks.
Both instances have shown IoU values of over 93%, which is more than enough to get a reasonable output style. Creating own masks
manually is very time-consuming, but I wanted to go through this process myself to test the output.

# Style transfer:
Experimented with different emphasis on content and style. As the objects seem to be less
recognisable in contrast to their surroundings without style, I have reduced the style_score
and increased the content_score in the calculation of loss. This resulted in better looking images.
Further experimentation can be done by adjusting the weights. Also, a user study can be conducted to
measure how well the style of the image is translated into the objects.


### How to run:
By executing the python files in virtart, the checkpoints are downloaded. The file path needs to be set accordingly. (~4gb!)
Download the requirements specified in requirements.txt. **The Python version of this project: 3.9.7**

In the notebooks you can see functionality of the pipeline step by step. (yolo detection -> SAM -> style)

# Work Breakdown Structure:
### Project setup:
- Estimated: 7h
- Actual: 12h

### Design and Build pipeline:
- Estimated: 20h
- Actual: 20h

### Fine-Tuning:
- Estimated: 17h
- Actual: 17h

### Application-Development:
- Estimated: 6h
- Actual: tbd

### Presentation preparation:
- Estimated: 3h
- Actual: tbd

### Further possible Todo's

- improve style transfer with adjusting weight etc.
- plot graph of loss with varying influences
- fix resolution issue by dividing the image in multiple patches -> Could be hard to combine image back together