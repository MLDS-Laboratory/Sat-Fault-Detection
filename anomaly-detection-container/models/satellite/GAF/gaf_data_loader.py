import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from gaf_transform import compute_gaf  
import matplotlib.pyplot as plt
import argparse
import os
import time
from functools import wraps
import pickle

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper


class GAFDataset(Dataset):
    """
    Torch Dataset that converts each OPS-SAT segment into a GAF image.
    By using torchvision transforms, setup to feed into CNN
    """
    def __init__(self, segments, transform=None, image_size=224, cache_dir=None):
        """
        Parameters:
        - segments: list of dicts with keys 'segment', 'channel', 'ts', 'label', 'sampling', 'train'
        - transform: torchvision transforms to apply to the GAF image
        - image_size: size of the output GAF image (default 224)
        - cache_dir: directory to cache GAF images. If None, caching is disabled.
        """
        self.segments = segments
        self.transform = transform
        self.image_size = image_size
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg_dict = self.segments[idx]
        ts = seg_dict['ts']
        label = seg_dict['label']

        # Try to load from cache first
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"gaf_{seg_dict['segment']}_{seg_dict['channel']}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    img = pickle.load(f)
            else:
                img = self._compute_gaf_image(ts)
                with open(cache_path, 'wb') as f:
                    pickle.dump(img, f)
        else:
            img = self._compute_gaf_image(ts)
            
        if self.transform:
            img = self.transform(img)
            
        return img, label
        
    def _compute_gaf_image(self, ts):
        # GAF transformation code
        gaf_img = compute_gaf(ts)
        gaf_img = (gaf_img - gaf_img.min()) / (gaf_img.max() - gaf_img.min() + 1e-8)
        gaf_img = np.uint8(255 * gaf_img)
        img = Image.fromarray(gaf_img).resize((self.image_size, self.image_size))
        return img.convert("RGB")