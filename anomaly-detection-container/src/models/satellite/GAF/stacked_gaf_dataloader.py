import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from models.satellite.GAF.gaf_transform import compute_gaf
from sklearn.model_selection import train_test_split


def stacked_stratified_sample(train_segs, test_segs, max_train_samples, max_test_samples, min_anomaly_pct=0.05, random_state=42):
    """
    Stratified sampler for Stacked GAFs. Ensures the training set has a minimum 
    percentage of anomalies, oversampling if necessary.
    """
    rng = np.random.default_rng(random_state)
    
    # 1. Sample Test Set (preserve original distribution)
    if len(test_segs) > max_test_samples:
        y_test = [s['label'] for s in test_segs]
        idx = np.arange(len(test_segs))
        keep, _ = train_test_split(idx, train_size=max_test_samples, stratify=y_test, random_state=random_state)
        test_segs = [test_segs[i] for i in keep]
        print(f"Sampled test set to {len(test_segs)} stacked segments.")

    # 2. Sample Train Set
    y_train = np.array([s['label'] for s in train_segs])
    anom_idx = np.where(y_train == 1)[0]
    norm_idx = np.where(y_train == 0)[0]
    
    target_anom = int(max_train_samples * min_anomaly_pct)
    target_norm = max_train_samples - target_anom
    
    # Sample Anomalies
    if len(anom_idx) == 0:
        sampled_anom = [] 
    else:
        replace_anom = len(anom_idx) < target_anom
        sampled_anom = rng.choice(anom_idx, target_anom, replace=replace_anom).tolist()
        
    # Sample Nominals
    if len(norm_idx) == 0:
        sampled_norm = []
    else:
        replace_norm = len(norm_idx) < target_norm
        sampled_norm = rng.choice(norm_idx, target_norm, replace=replace_norm).tolist()
        
    final_idx = sampled_anom + sampled_norm
    rng.shuffle(final_idx)
    
    train_segs = [train_segs[i] for i in final_idx]
    
    anom_share = sum(1 for s in train_segs if s['label'] == 1) / len(train_segs) * 100 if train_segs else 0
    print(f"Sampled train set to {len(train_segs)} stacked segments. Anomaly share: {anom_share:.2f}%")
    
    return train_segs, test_segs

class StackedGAFDataset(Dataset):
    def __init__(self, segments, image_size=224, cache_dir=None):
        self.segments = segments
        self.image_size = image_size
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg_dict = self.segments[idx]
        ts_2d = seg_dict['ts'] # Shape: (Seq_len, Channels)
        label = seg_dict['label']
        seg_id = seg_dict['segment']

        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"stacked_gaf_{seg_id}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    tensor_img = pickle.load(f)
                return tensor_img, label

        # OPTIMIZATION: Downsample the 1D time series FIRST.
        # 1. Convert to tensor and reshape to (Batch=1, Channels, Seq_len)
        ts_tensor = torch.from_numpy(ts_2d.copy()).float().t().unsqueeze(0)
        
        # 2. Interpolate the time series down to exactly `image_size` (e.g., 224 points)
        ts_downsampled = F.interpolate(ts_tensor, size=self.image_size, mode='linear', align_corners=False)
        ts_downsampled = ts_downsampled.squeeze(0).numpy() # Shape: (Channels, 224)

        # 3. Compute GAFs. Because input is 224, output is natively 224x224!
        num_channels = ts_downsampled.shape[0]
        gafs = []
        for c in range(num_channels):
            gaf = compute_gaf(ts_downsampled[c, :])
            gaf = (gaf - gaf.min()) / (gaf.max() - gaf.min() + 1e-8)
            gafs.append(gaf)
            
        stacked = np.stack(gafs, axis=0) # Shape: (Channels, 224, 224)
        tensor_img = torch.from_numpy(stacked).float()
        
        # Normalize: mean=0.5, std=0.5 across all channels
        tensor_img = (tensor_img - 0.5) / 0.5

        if self.cache_dir:
            with open(cache_path, 'wb') as f:
                pickle.dump(tensor_img, f)

        return tensor_img, label