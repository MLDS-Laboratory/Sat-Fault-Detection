import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from gaf_transform import compute_gaf  
import matplotlib.pyplot as plt
import argparse
import os


class OpsSatGAFDataset(Dataset):
    """
    Torch Dataset that converts each OPS-SAT segment into a GAF image.
    By using torchvision transforms, setup to feed into CNN
    """
    def __init__(self, segments, transform=None, image_size=224):
        self.segments = segments
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg_dict = self.segments[idx]
        ts = seg_dict['ts']
        label = seg_dict['label']

        # compute the gaf
        gaf_img = compute_gaf(ts)

        # normalize the GAF image values to [0, 1]
        gaf_img = (gaf_img - gaf_img.min()) / (gaf_img.max() - gaf_img.min() + 1e-8)
        gaf_img = np.uint8(255 * gaf_img)

        # convert to PIL image and resize
        img = Image.fromarray(gaf_img).resize((self.image_size, self.image_size))

        # convert to 3 channel RGB to work with existing CNNs
        # TODO: check if this is necessary and if it hurts the fully-trained CNNs
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label