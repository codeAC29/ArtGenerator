import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from parameters import *
from image_loader import ImageFolder

def get_dataset():
    transform = transforms.Compose([transforms.Resize(image_size+10),
                                     transforms.CenterCrop(image_size),
                                     #transforms.RandomCrop(image_size),
                                     #transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImageFolder(root=dataroot,transform=transform)

    return DataLoader(dataset, batch_size, shuffle=True, num_workers=workers, pin_memory=True)
