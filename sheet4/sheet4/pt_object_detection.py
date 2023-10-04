#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision.io.image import read_image
# reads and converts it to a pytorch tensor

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_resnet50_fpn,

)
# These are the Faster R-CNN model architecture and pre-trained weights for the MobileNetV3 backbone.

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image 
from torchvision.transforms import Compose, ConvertImageDtype, PILToTensor
from PIL import Image

#This function converts a PyTorch tensor to a PIL (Python Imaging Library) image.


import matplotlib.pyplot as plt


# TODO: Load weights and construct the model with them and a box_score_thresh of 0.3
    # Tipp: The weights are available as FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    #       and the model is available as fasterrcnn_mobilenet_v3_large_fpn in torchvision.models.detection

#load the weights
weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT

#construct the model
model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
#model = fasterrcnn_resnet50_fpn(pretrained=True)

model.box_score_thresh = 0.3
#

# TODO: Set the model to evaluation mode (to disable dropout and batch normalization)
model.eval()
# In evaluation mode, the model disables dropout regularization and uses the running statistics of batch normalization layers instead of
# computing batch statistics. This ensures consistent and deterministic behavior during inference.
#

# TODO: Load image and apply preprocessing transforms
# image path: "img/TUBraunschweig_Universitaetsplatz.jpg" source: https://www.tu-braunschweig.de/
# Tipp: The model expects a tensor - you can use the transforms() from the loaded weights to convert the image
#       The model expects a list of images, so wrap the image in a list

img_src = "C:\Vicky\Tu Braunschweig\Semester 2\Computer vision\Exercise\Team13\sheet4\sheet4\img\TUBraunschweig_Universitaetsplatz.jpg"
#img = read_image(img_src)
#img = Image.open(img_src)

# Preprocessing
img = Image.open(img_src).convert("RGB")
# Preprocessing
transform = Compose([
    PILToTensor(),
    ConvertImageDtype(torch.float32),
])

input_tensor = transform(img).unsqueeze(0)
print(input_tensor.shape)
print(input_tensor.min(), input_tensor.max())

#alternate preprocessing
# img = read_image(data_dir)
# preprocess = weights.transforms()
# input_tensor = [preprocess(img)]


## TODO: Use the model and visualize the prediction using matplotlib
# Tipp: The model returns a dict-like structure with the predictions
#       The model returns a batch of predictions, we only need the first one for our single image
#
predictions = model(input_tensor)[0]

## TODO: Visualize the prediction using the draw_bounding_boxes() function and matplotlib
labels = [weights.meta["categories"][i] for i in predictions["labels"]]
img_int = torch.tensor(input_tensor*255, dtype=torch.uint8)

img_boxes = draw_bounding_boxes(image=img_int[0],
                          boxes=predictions["boxes"],
                          labels=labels,
                          colors="red",
                          width=2
                          )
plt.imshow(to_pil_image(img_boxes.detach()))
plt.show()

