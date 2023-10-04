#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#
# Task 3
#
# Create a network to recognize single handwritten digits (0-9)


import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


# TODO: Define transformation to normalize all input data to [0, 1].
#  (https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html)

transform = transform = transforms.ToTensor()
#
# ???
#

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

size_trainset = len(trainset)

print(f"Number of training samples: {size_trainset}")
print(f"Number of test samples: {len(testset)}")

# ---------------------------------------------------------------------------------------------


# Define Network architecture
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO: Define the modules of your network:
        # flatten
        # linear (128 outputs)
        # relu
        # linear (10 outputs)
        # softmax

        #   
        # ???
        #

    def forward(self, x):
        # TODO: Define the forward pass of your network
        #
        # ???
        #


# TODO: Initialize the model, optimizer (SGD) and loss function (CrossEntropyLoss)
model = ...
#
# ???
#

# check the structure of the model
print(model)

# ---------------------------------------------------------------------------------------------

# Train the network
for epoch in range(10):  # loop over the dataset multiple times
    print(f"Epoch {epoch}:")
    # iterate over the batches of the dataset
    for batch, (inputs, labels) in enumerate(trainloader):
        # TODO: forward + loss + backward + optimize
        #
        # ???
        #

        # TODO: reset the parameter gradients
        #
        # ???
        #

        # print statistics
        if batch % 100 == 0:
            current = (batch + 1) * len(inputs)
            print(f"loss: {loss.item():>5f}  [{current:>5d}/{size_trainset:>5d}]")

print("Finished Training")

# ---------------------------------------------------------------------------------------------

# Use the trained model to recognize the digit of some test set images
num_samples = 10
sample_images = next(iter(testloader))[0][:num_samples]

outputs = model(sample_images)
probs, predicted_class = torch.max(outputs, 1)

fig, axs = plt.subplots(1, num_samples, figsize=(20, 2))
for i, ax in enumerate(axs):
    ax.imshow(sample_images[i].permute(1, 2, 0), cmap="gray")
    ax.set_xlabel(f"Predicted: {predicted_class[i]}, \nProb: {probs[i]:.3f}")
    ax.xaxis.label.set_size(12)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

plt.show()
