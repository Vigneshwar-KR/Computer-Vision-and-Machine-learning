import time
import torch
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from PIL import Image

from flow_utils import load_FLO_file, flowMapToBGR, drawArrows, calculateAngularError
from utils import showImages
import cv2
import matplotlib.pyplot as plt

from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large


#
# Task 3
#
# Load and use a pretrained model to estimate the optical flow of the same two frames as in Task 2.



# Load image frames
img1_batch = torch.stack([read_image("C:/Vicky/Tu Braunschweig/Semester 2/Computer vision/Exercise/Team13/sheet5/sheet5/resources/frame1.png")])
img2_batch = torch.stack([read_image("C:/Vicky/Tu Braunschweig/Semester 2/Computer vision/Exercise/Team13/sheet5/sheet5/resources/frame2.png")])

# Convert images to tensors
img1_tensor = (img1_batch).unsqueeze(0)
img2_tensor = (img2_batch).unsqueeze(0)

# Load ground truth flow data for evaluation
flow_gt = load_FLO_file("C:/Vicky/Tu Braunschweig/Semester 2/Computer vision/Exercise/Team13/sheet5/sheet5/resources/groundTruthOF.flo")


# TODO: Load the model weights and prepare the images for inference.
weights = Raft_Large_Weights.DEFAULT
preprocess = weights.transforms()
img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)

#


# TODO: Remember to apply the same preprocessing to the ground truth flow in regards to image size.
#
#flow_gt = preprocess(flow_gt)  
#


# TODO: Load the model
#
model.eval()

#


# TODO: Compute the optical flow using the model
#
list_of_flows = model(img1_batch, img2_batch)
#


# TODO Create and show visualizations for the computed flow
#
flow_vis = flowMapToBGR(list_of_flows)
flow_arrows = drawArrows(img1_batch, list_of_flows)
#


