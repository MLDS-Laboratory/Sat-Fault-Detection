import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from models.satellite.GAF.gaf_transform import compute_gaf
import os
import time
from functools import wraps
from math import ceil
import pickle
from sklearn.model_selection import train_test_split

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper

def stratified_sample(train_segs, test_segs, max_train_samples, max_test_samples, *,
        oversample_anomaly=True, min_anomaly_pct=0.05, random_state=42):
    """
    Stratified sampling for train / test sets.

    Parameters
    ----------
    train_segs, test_segs : list[dict]
        Segment dictionaries that contain at least fields
        - 'label'   : 1 = anomaly, 0 = nominal
        - 'channel' : telemetry channel id / name
    max_train_samples, max_test_samples : int
        Upper bounds on the size of the returned splits.
    oversample_anomaly : bool, default True
        If True, the returned *train* split is forced to contain at least
        `min_anomaly_pct` anomalies (duplicates allowed when the corpus
        does not have enough unique anomalies).
    min_anomaly_pct : float
        Desired minimum anomaly share for the training set.
    random_state : int, default 42
        Reproducible RNG seed.

    Returns
    -------
    train_segs_out, test_segs_out : list[dict]
    """
    rng = np.random.default_rng(random_state)

    # -------------------- 1. SAMPLE TEST SET (leave distribution intact) ---
    if len(test_segs) > max_test_samples:
        y_test = [seg['label'] for seg in test_segs]
        idx    = np.arange(len(test_segs))
        keep, _ = train_test_split(
            idx,
            test_size=(len(test_segs) - max_test_samples) / len(test_segs),
            stratify=y_test,
            random_state=random_state
        )
        test_segs = [test_segs[i] for i in keep]
        print(f"Sampled {len(test_segs)} test segments using stratified sampling")

    # -------------------- 2. SAMPLE TRAIN SET --------------------------------
    # Case A – just size‑limit with label‑stratification 
    if not oversample_anomaly or len(train_segs) <= max_train_samples:
        if len(train_segs) > max_train_samples:
            y_train = [seg['label'] for seg in train_segs]
            idx     = np.arange(len(train_segs))
            keep, _ = train_test_split(
                idx,
                test_size=(len(train_segs) - max_train_samples) / len(train_segs),
                stratify=y_train,
                random_state=random_state
            )
            train_segs = [train_segs[i] for i in keep]
            print(f"Sampled {len(train_segs)} train segments using stratified sampling")
        return train_segs, test_segs

    # Case B – enforce a minimum anomaly share in the training split
    rng = np.random.default_rng(random_state)

    target_anom = int(ceil(min_anomaly_pct * max_train_samples))
    target_norm = max_train_samples - target_anom

    # Build per‑channel index pools
    chan_idx = {}
    for i, seg in enumerate(train_segs):
        ch = seg['channel']
        if ch not in chan_idx:
            chan_idx[ch] = {'anom': [], 'norm': []}
        key = 'anom' if seg['label'] == 1 else 'norm'
        chan_idx[ch][key].append(i)

    #  helper to sample a list of indices, always returned as int
    def _sample(idx_list, k, replace):
        """Return *k* int indices sampled from idx_list."""
        if k <= 0:
            return []
        arr = rng.choice(idx_list, k, replace=replace)
        # rng.choice returns ndarray; ensure pure python ints
        return [int(x) for x in (arr.tolist() if hasattr(arr, 'tolist') else [arr])]

    #  sample anomalies (keep channel proportions) 
    all_anom_idx = np.concatenate([v['anom'] for v in chan_idx.values()]).astype(int)
    sampled_anom = []

    if len(all_anom_idx) >= target_anom:
        # proportional per channel
        for ch, pools in chan_idx.items():
            n_ch = len(pools['anom'])
            if n_ch == 0:
                continue
            k = int(round(n_ch / len(all_anom_idx) * target_anom))
            k = min(k, n_ch)  # guard
            sampled_anom.extend(_sample(pools['anom'], k, replace=False))

        # fill any shortfall (due to rounding) at random
        deficit = target_anom - len(sampled_anom)
        if deficit > 0:
            remaining = list(set(all_anom_idx) - set(sampled_anom))
            sampled_anom.extend(_sample(remaining, deficit, replace=False))
    else:
        # not enough uniques → oversample with replacement
        sampled_anom.extend(_sample(all_anom_idx, target_anom, replace=True))

    #  sample normals 
    all_norm_idx = np.concatenate([v['norm'] for v in chan_idx.values()]).astype(int)
    replace_norm = len(all_norm_idx) < target_norm
    sampled_norm = _sample(all_norm_idx, target_norm, replace=replace_norm)

    #  assemble & shuffle 
    final_idx = sampled_anom + sampled_norm
    rng.shuffle(final_idx)

    train_segs = [train_segs[i] for i in final_idx]

    anom_share = 100 * sum(seg['label'] for seg in train_segs) / len(train_segs)
    print(f"Sampled {len(train_segs)} train segments "
          f"(anomaly share: {anom_share:.2f} %, target ≥ {min_anomaly_pct*100:.0f} %)")

    return train_segs, test_segs


import torch

def mil_collate(batch):
    """
    Custom collate function for MIL. 
    Returns bags as a list, labels as a tensor, and event_ids as a tensor.
    """
    bags = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    event_ids = torch.tensor([item[2] for item in batch])
    return bags, labels, event_ids

class GAFDataset(Dataset):
    """
    Torch Dataset that converts each ESA segment bag into a stack of GAF images.
    Implements MIL by returning [N, C, H, W] where N is bag size.
    """
    def __init__(self, segments, transform=None, image_size=224, cache_dir=None):
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
        bag_ts = seg_dict['ts']  # This is now a list of arrays
        label = seg_dict['label']
        event_id = seg_dict.get('event_id', -1)

        # Manual cache versioning - edit this string only if the cache must be invalidated
        CACHE_VERSION = "v1" 

        bag_imgs = []
        for i, ts in enumerate(bag_ts):
            # Unique cache key per window in the bag
            ch_key = seg_dict.get('channel', 'stacked')
            cache_path = None
            if self.cache_dir:
                cache_path = os.path.join(self.cache_dir, f"gaf_{seg_dict['segment']}_{ch_key}_bag{i}_{CACHE_VERSION}.pkl")
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
            
            # Ensure img is a tensor
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            elif not isinstance(img, torch.Tensor):
                img = torch.tensor(img, dtype=torch.float32)
                
            bag_imgs.append(img)
            
        return torch.stack(bag_imgs), label, event_id
        
    def _compute_gaf_image(self, ts):
        # Optimized: Downsample the 1D signal to image_size BEFORE O(N^2) GAF computation
        def downsample_1d(arr, n):
            if len(arr) == n: return arr
            return np.interp(np.linspace(0, len(arr)-1, n), np.arange(len(arr)), arr)

        # Handle 1D (Single Channel) or 2D (Stacked)
        if ts.ndim == 1:
            ts_resampled = downsample_1d(ts, self.image_size)
            gaf_img = compute_gaf(ts_resampled)
            gaf_img = (gaf_img - gaf_img.min()) / (gaf_img.max() - gaf_img.min() + 1e-8)
            gaf_img = np.uint8(255 * gaf_img)
            img = Image.fromarray(gaf_img).resize((self.image_size, self.image_size)).convert("L")
        else:
            # Stacked Case: ts is [SeqLen, NumChannels]
            num_channels = ts.shape[1]
            gafs = []
            for c in range(num_channels):
                ts_c = downsample_1d(ts[:, c], self.image_size)
                g = compute_gaf(ts_c)
                g = (g - g.min()) / (g.max() - g.min() + 1e-8)
                gafs.append(np.uint8(255 * g))
            
            # Combine into a multi-channel image
            combined = np.stack(gafs, axis=0) # [C, H, W]
            # Resize each channel (already image_size, but kept for consistency)
            resized_gafs = []
            for c in range(num_channels):
                img_c = Image.fromarray(combined[c]).resize((self.image_size, self.image_size))
                resized_gafs.append(np.array(img_c))
            img = np.stack(resized_gafs, axis=0) # [C, H, W]
            return img 

        return img