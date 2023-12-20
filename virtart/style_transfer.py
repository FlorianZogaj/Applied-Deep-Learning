import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

# using neural style transfer tutorial https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    # print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img,
                                                                     content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    # print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score * 0.3 + content_score * 5
            loss.backward()

            run[0] += 1
            # if run[0] % 50 == 0:
            #     print("run {}:".format(run))
            #     print('Style Loss : {:4f} Content Loss: {:4f}'.format(
            #         style_score.item(), content_score.item()))
            #     print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    final_style_loss = sum([sl.loss.item() for sl in style_losses])
    return input_img, final_style_loss

    # return input_img


def adjust_bounds_and_translate(y_min, y_max, x_min, x_max, array, array_height=480, array_width=640):
    # Ensure the minimum bounds are not negative
    y_min_adj = max(y_min, 0)
    x_min_adj = max(x_min, 0)

    # Trim the beginning of the array if necessary
    array = array[y_min_adj - y_min:, x_min_adj - x_min:]

    # Ensure the maximum bounds do not exceed the array dimensions
    y_max_adj = min(y_max - y_min + y_min_adj, array_height)
    x_max_adj = min(x_max - x_min + x_min_adj, array_width)

    # Trim the end of the array if necessary
    array = array[:y_max_adj - y_min_adj, :x_max_adj - x_min_adj]

    return array


class StyleTransfer:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(self.device)
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

    def crop_and_resize(self, frame, x_min, x_max, y_min, y_max, resize=(512, 512)):
        cropped_frame = frame[max(y_min, 0):min(frame.shape[0], y_max),
                        max(x_min, 0):min(frame.shape[1], x_max)]
        pad_left = max(0, -x_min)
        pad_right = max(0, x_max - frame.shape[1])
        pad_top = max(0, -y_min)
        pad_bottom = max(0, y_max - frame.shape[0])
        if len(frame.shape) == 3:
            padded_frame = np.pad(cropped_frame, ((
                                                      pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')
        else:
            padded_frame = np.pad(
                cropped_frame, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
        return cv2.resize(padded_frame, resize)

    # backround stlye transfer will be further explored in future
    def background(self, img, mask, style):
        loader = transforms.Compose([
            transforms.Resize((256, 256)),  # scale imported image
            transforms.ToTensor()]
        )  # transform it into a torch tensor

        content_tensor = loader(Image.fromarray(img)).unsqueeze(0).to(self.device)
        style_tensor = loader(style).unsqueeze(0).to(self.device)

        input_tensor = content_tensor.clone()
        output_tensor = run_style_transfer(
            self.cnn, self.cnn_normalization_mean, self.cnn_normalization_std,
            content_tensor, style_tensor, input_tensor
        )

        with torch.no_grad():
            output_tensor_right_size = F.interpolate(output_tensor, size=(
                img.shape[0], img.shape[1]), mode='bilinear', align_corners=False)

        output_right_size = output_tensor_right_size.squeeze().cpu().numpy()
        output_right_size = output_right_size.transpose((1, 2, 0)).copy()
        output_right_size = (output_right_size * 255).astype(np.uint8)

        # translated_mask = mask[y_min:y_max, x_min:x_max]
        return output_right_size, mask.copy()

    def __call__(self, img, mask, style):
        y_indices, x_indices = np.where(mask)

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Add padding of 1.2 to bounding box
        x_padding = int((x_max - x_min) * 0.1)
        y_padding = int((y_max - y_min) * 0.1)

        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        side_length = max(x_max - x_min, y_max - y_min)
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        x_min, x_max = center_x - side_length // 2, center_x + side_length // 2
        y_min, y_max = center_y - side_length // 2, center_y + side_length // 2

        cropped_img = self.crop_and_resize(img, x_min, x_max, y_min, y_max)

        loader = transforms.Compose([
            transforms.Resize((512, 512)),  # scale imported image
            transforms.ToTensor()]
        )  # transform it into a torch tensor

        content_tensor = loader(Image.fromarray(cropped_img)).unsqueeze(0).to(self.device)
        style_tensor = loader(style).unsqueeze(0).to(self.device)

        input_tensor = content_tensor.clone()
        output_tensor, final_style_loss = run_style_transfer(
            self.cnn, self.cnn_normalization_mean, self.cnn_normalization_std,
            content_tensor, style_tensor, input_tensor
        )

        with torch.no_grad():
            x_min = int(x_min)
            x_max = int(x_max)
            y_min = int(y_min)
            y_max = int(y_max)
            output_tensor_right_size = F.interpolate(output_tensor, size=(
                y_max - y_min, x_max - x_min), mode='bilinear', align_corners=False)
            output_tensor_right_size = adjust_bounds_and_translate(
                y_min, y_max, x_min, x_max, output_tensor_right_size,
                array_height=mask.shape[0], array_width=mask.shape[1]
            )
        output_right_size = output_tensor_right_size.squeeze().cpu().numpy()
        output_right_size = output_right_size.transpose((1, 2, 0)).copy()
        output_right_size = (output_right_size * 255).astype(np.uint8)

        translated_mask = mask[y_min:y_max, x_min:x_max]
        return output_right_size, translated_mask, final_style_loss
