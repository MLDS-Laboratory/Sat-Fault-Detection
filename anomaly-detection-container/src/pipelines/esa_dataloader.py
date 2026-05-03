import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class ESAMissionDataLoader:
    """
    Fixed-Duration Window Data Loader for ESA Mission data.
    Implements Multiple Instance Learning (MIL) by splitting long anomalies into 
    multiple windows and padding short ones.
    """

    def __init__(self, mission_dir: str, nominal_segment_len: int | None = None, train_ratio: float = 0.8, random_state: int = 42):
        self.dir = os.path.abspath(mission_dir)
        self.nominal_segment_len = nominal_segment_len
        self.train_ratio = train_ratio
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # internal
        self.channels_df: pd.DataFrame | None = None
        self.labels_df: pd.DataFrame | None = None
        self.segments: list[dict] | None = None
        self.fixed_duration_sec: float = 0.0
        self.target_sampling_sec: float = 0.0

    def _load_meta(self):
        cfile = os.path.join(self.dir, "channels.csv")
        lfile = os.path.join(self.dir, "labels.csv")

        self.channels_df = pd.read_csv(cfile)
        self.labels_df = pd.read_csv(lfile)

        self.labels_df["StartTime"] = pd.to_datetime(self.labels_df["StartTime"])
        self.labels_df["EndTime"] = pd.to_datetime(self.labels_df["EndTime"])

        # Calculate Anomaly Duration Statistics
        durations = (self.labels_df["EndTime"] - self.labels_df["StartTime"]).dt.total_seconds()
        self.fixed_duration_sec = durations.mean()
        
        print(f"\n--- Anomaly Duration Statistics ---")
        print(f"Mean: {self.fixed_duration_sec:.2f}s")
        print(f"Std:  {durations.std():.2f}s")
        print(f"Max:  {durations.max():.2f}s")
        print(f"Min:  {durations.min():.2f}s")
        print(f"Fixed Window Duration set to: {self.fixed_duration_sec:.2f}s")

    def _build_segments(self):
        self._load_meta()
        
        # Mapping for physical event IDs to integers
        unique_ids = self.labels_df["ID"].unique().tolist()
        id_map = {id_str: i for i, id_str in enumerate(unique_ids)}

        # 1. Determine target sampling rate (median across all channels)
        intervals = []
        ch_dir = os.path.join(self.dir, "channels")
        for _, ch_row in self.channels_df.iterrows():
            ch_name = ch_row["Channel"]
            pkl_path = os.path.join(ch_dir, ch_name)
            if os.path.isfile(pkl_path):
                ts_df = pd.read_pickle(pkl_path)
                interval = ts_df.index.to_series().diff().dt.total_seconds().median()
                if pd.notna(interval): intervals.append(interval)
        
        self.target_sampling_sec = np.median(intervals)
        print(f"Target Sampling Interval: {self.target_sampling_sec:.2f}s")
        
        segments = []
        seg_id = 0

        for _, ch_row in self.channels_df.iterrows():
            ch_name = ch_row["Channel"]
            pkl_path = os.path.join(ch_dir, ch_name)
            if not os.path.isfile(pkl_path): continue

            ts_df: pd.DataFrame = pd.read_pickle(pkl_path).sort_index()
            # Resample to target frequency for consistency
            ts_df.index = ts_df.index.tz_localize(None)
            ts_resampled = ts_df.resample(f"{int(self.target_sampling_sec)}s").mean().interpolate(method='linear').dropna()
            
            idx = ts_resampled.index
            values = ts_resampled.iloc[:, 0].astype(np.float32).values
            
            ch_labels = self.labels_df[self.labels_df["Channel"] == ch_name]

            # ------------------------------------------------------------------
            # 2. Extract Anomalous Segments (with MIL splitting/padding)
            # ------------------------------------------------------------------
            for _, lab in ch_labels.iterrows():
                start, end = lab["StartTime"].tz_localize(None), lab["EndTime"].tz_localize(None)
                event_dur = (end - start).total_seconds()
                
                bag_ts = []
                
                if event_dur <= self.fixed_duration_sec:
                    # Case A: Short Anomaly -> Centered Padding
                    center_time = start + (end - start) / 2
                    win_start = center_time - pd.Timedelta(seconds=self.fixed_duration_sec / 2)
                    win_end = win_start + pd.Timedelta(seconds=self.fixed_duration_sec)
                    
                    mask = (idx >= win_start) & (idx <= win_end)
                    if mask.any():
                        bag_ts.append(values[mask])
                else:
                    # Case B: Long Anomaly -> Randomized MIL Splitting
                    # Calculate number of windows needed to cover the span
                    n_windows = int(np.ceil(event_dur / self.fixed_duration_sec))
                    total_coverage = n_windows * self.fixed_duration_sec
                    slack = total_coverage - event_dur
                    
                    # Randomly distribute the slack at the start
                    initial_offset = self.rng.uniform(0, slack)
                    current_start = start - pd.Timedelta(seconds=initial_offset)
                    
                    for _ in range(n_windows):
                        win_end = current_start + pd.Timedelta(seconds=self.fixed_duration_sec)
                        mask = (idx >= current_start) & (idx <= win_end)
                        if mask.any():
                            bag_ts.append(values[mask])
                        current_start = win_end - pd.Timedelta(seconds=slack/(n_windows-1) if n_windows > 1 else 0)

                if bag_ts:
                    # Filter out any windows that are significantly shorter than expected due to edges
                    target_pts = int(self.fixed_duration_sec / self.target_sampling_sec)
                    valid_bag = [ts for ts in bag_ts if abs(len(ts) - target_pts) < 5]
                    if valid_bag:
                        segments.append({
                            "segment": seg_id, "channel": ch_name, "ts": valid_bag, 
                            "label": 1, "sampling": self.target_sampling_sec, "train": 1,
                            "event_id": id_map[lab["ID"]]
                        })
                        seg_id += 1

            # ------------------------------------------------------------------
            # 3. Extract Nominal Segments (Fixed Blocks)
            # ------------------------------------------------------------------
            mask_anom = np.zeros(len(idx), dtype=bool)
            for _, lab in ch_labels.iterrows():
                s, e = lab["StartTime"].tz_localize(None), lab["EndTime"].tz_localize(None)
                mask_anom |= (idx >= s) & (idx <= e)

            nominal_idx = np.where(~mask_anom)[0]
            if len(nominal_idx) == 0: continue
            
            blocks = np.split(nominal_idx, np.where(np.diff(nominal_idx) != 1)[0] + 1)
            target_pts = int(self.fixed_duration_sec / self.target_sampling_sec)

            for blk in blocks:
                for i in range(0, len(blk) - target_pts, target_pts):
                    seg_slice = blk[i : i + target_pts]
                    segments.append({
                        "segment": seg_id, "channel": ch_name, "ts": [values[seg_slice]], 
                        "label": 0, "sampling": self.target_sampling_sec, "train": 1,
                        "event_id": -(seg_id + 1)
                    })
                    seg_id += 1

        self.segments = segments

    def get_train_val_test_segments(self, train_ratio: float = 0.64, val_ratio: float = 0.16, test_ratio: float = 0.20):
        if self.segments is None: self._build_segments()
        
        anom_segs = [s for s in self.segments if s['label'] == 1]
        norm_segs = [s for s in self.segments if s['label'] == 0]
        
        self.rng.shuffle(anom_segs)
        self.rng.shuffle(norm_segs)
        
        def split_list(lst, r1, r2):
            n = len(lst)
            idx1 = int(r1 * n)
            idx2 = int((r1 + r2) * n)
            return lst[:idx1], lst[idx1:idx2], lst[idx2:]

        tr_anom, val_anom, te_anom = split_list(anom_segs, train_ratio, val_ratio)
        tr_norm, val_norm, te_norm = split_list(norm_segs, train_ratio, val_ratio)

        train = tr_anom + tr_norm
        val   = val_anom + val_norm
        test  = te_anom + te_norm

        self.rng.shuffle(train)
        self.rng.shuffle(val)
        self.rng.shuffle(test)

        print(f"Event-aware splits created: "
              f"Train={len(train)} (anom={len(tr_anom)}), "
              f"Val={len(val)} (anom={len(val_anom)}), "
              f"Test={len(test)} (anom={len(te_anom)})")
        
        if len(tr_anom) < 10 or len(val_anom) < 10 or len(te_anom) < 10:
             print("WARNING: One or more splits contains fewer than 10 anomaly events. Consider increasing sample budget.")

        return train, val, test

