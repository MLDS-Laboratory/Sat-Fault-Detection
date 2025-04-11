import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from gaf_transform import compute_gaf  
import matplotlib.pyplot as plt
import argparse
import os

class OpsSatDataLoader:
    """
    Class to load the OPS-SAT data.
    Reads dataset.csv and segment.csv files and creates a list of segments
    with associated metadata.
    """
    def __init__(self, dataset_csv, segment_csv):
        self.dataset_csv = dataset_csv
        self.segment_csv = segment_csv
        self.meta_df = None
        self.seg_df = None

    def load_data(self):
        self.meta_df = pd.read_csv(self.dataset_csv)
        self.seg_df = pd.read_csv(self.segment_csv)

    def get_segments(self):
        # Group time-series data by segment
        grouped = self.seg_df.groupby('segment')
        segments = []
        for seg, group in grouped:
            group = group.sort_values('timestamp')
            ts = group['value'].values.astype(np.float32)
            anomaly_flag = int(group['anomaly'].iloc[0])

            channel = group['channel'].iloc[0]
            sampling = group['sampling'].iloc[0]
            train_flag = group['train'].iloc[0]
            segments.append({
                'segment': seg,
                'channel': channel,
                'ts': ts,
                'label': anomaly_flag,
                'sampling': sampling,
                'train': train_flag
            })
        return segments

    def get_train_test_segments(self):
        if self.meta_df is None or self.seg_df is None:
            self.load_data()
        segments = self.get_segments()
        # Split segments using the "train" flag
        train_data = [s for s in segments if s['train'] == 1]
        test_data = [s for s in segments if s['train'] == 0]
        return train_data, test_data


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

if __name__ == "__main__":
    dataset_csv = "C:\\Users\\varun\\Documents\\UMD\\Research\\MLDS\\Sat-Fault-Detection\\anomaly-detection-container\\data\\OPS-SAT\\dataset.csv"
    segment_csv = "C:\\Users\\varun\\Documents\\UMD\\Research\\MLDS\\Sat-Fault-Detection\\anomaly-detection-container\\data\\OPS-SAT\\segments.csv"
    
    segment_idx = 0  # Default segment index to display

    # Print paths to help with debugging
    print(f"Loading dataset from: {dataset_csv}")
    print(f"Loading segments from: {segment_csv}")
    
    # Load data
    data_loader = OpsSatDataLoader(dataset_csv, segment_csv)
    train_segments, test_segments = data_loader.get_train_test_segments()
    
    # Combine segments for full range of indices
    all_segments = train_segments + test_segments
    
    if segment_idx < 0 or segment_idx >= len(all_segments):
        print(f"Error: segment_idx must be between 0 and {len(all_segments)-1}")
        exit(1)
    
    # get segment and create GAF image
    segment = all_segments[segment_idx]
    ts = segment['ts']
    label = segment['label']
    label_str = "Anomaly" if label == 1 else "Normal"
    channel = segment['channel']
    
    plt.figure(figsize=(12, 6))
    
    # Plot original time series
    plt.subplot(1, 2, 1)
    plt.plot(ts)
    plt.title(f"Original Time Series\nSegment {segment['segment']}, Channel: {channel}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    # Compute and display GAF
    gaf_img = compute_gaf(ts)
    
    # Normalize for visualization
    gaf_img = (gaf_img - gaf_img.min()) / (gaf_img.max() - gaf_img.min() + 1e-8)
    
    plt.subplot(1, 2, 2)
    plt.imshow(gaf_img, cmap='viridis')
    plt.colorbar()
    plt.title(f"GAF Image - Label: {label_str}")
    plt.tight_layout()
    
    print(f"Displaying segment {segment_idx} - ID: {segment['segment']}")
    print(f"Channel: {channel}, Label: {label_str}, Sampling: {segment['sampling']}")
    print(f"Time series length: {len(ts)}")
    
    plt.show()