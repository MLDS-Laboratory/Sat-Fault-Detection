import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ESAStackedDataLoader:
    def __init__(self, mission_dir: str, nominal_segment_len: int = 2048, train_ratio: float = 0.8, random_state: int = 42):
        self.dir = os.path.abspath(mission_dir)
        self.nominal_segment_len = nominal_segment_len
        self.train_ratio = train_ratio
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.segments = None

    def get_train_val_test_segments(self, train_ratio: float = 0.64, val_ratio: float = 0.16, test_ratio: float = 0.20):
        if self.segments is None: self._build_stacked_segments()
        
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

        print(f"[Stacked] Event-aware splits created: "
              f"Train={len(train)} (anom={len(tr_anom)}), "
              f"Val={len(val)} (anom={len(val_anom)}), "
              f"Test={len(test)} (anom={len(te_anom)})")
        
        if len(tr_anom) < 10 or len(val_anom) < 10 or len(te_anom) < 10:
             print("WARNING: One or more splits contains fewer than 10 anomaly events. Consider increasing sample budget.")

        return train, val, test

    def _build_stacked_segments(self):
        cfile = os.path.join(self.dir, "channels.csv")
        lfile = os.path.join(self.dir, "labels.csv")
        channels_df = pd.read_csv(cfile)
        labels_df = pd.read_csv(lfile)
        
        labels_df["StartTime"] = pd.to_datetime(labels_df["StartTime"])
        labels_df["EndTime"] = pd.to_datetime(labels_df["EndTime"])

        # Calculate Anomaly Duration Statistics
        durations = (labels_df["EndTime"] - labels_df["StartTime"]).dt.total_seconds()
        fixed_duration_sec = durations.mean()
        
        print(f"\n--- [Stacked] Anomaly Duration Statistics ---")
        print(f"Mean: {fixed_duration_sec:.2f}s")
        print(f"Std:  {durations.std():.2f}s")
        print(f"Max:  {durations.max():.2f}s")
        print(f"Min:  {durations.min():.2f}s")
        print(f"Fixed Window Duration set to: {fixed_duration_sec:.2f}s")

        ch_dir = os.path.join(self.dir, "channels")
        all_series = {}
        intervals = []

        for _, ch_row in channels_df.iterrows():
            ch_name = ch_row["Channel"]
            pkl_path = os.path.join(ch_dir, ch_name)
            if not os.path.isfile(pkl_path): continue
            
            ts_df = pd.read_pickle(pkl_path).sort_index()
            ts_df.index = ts_df.index.tz_localize(None) 
            all_series[ch_name] = ts_df.iloc[:, 0]
            
            interval = ts_df.index.to_series().diff().dt.total_seconds().median()
            if pd.notna(interval): intervals.append(interval)

        median_interval = np.median(intervals)
        global_freq = f"{int(median_interval)}s" if intervals else "1s"
        actual_pts = int(fixed_duration_sec / median_interval)
        
        df_all = pd.DataFrame(all_series)
        df_all = df_all.resample(global_freq).mean().interpolate(method='linear').dropna()
        
        anomaly_mask = np.zeros(len(df_all), dtype=bool)
        for _, lab in labels_df.iterrows():
            start, end = lab["StartTime"].tz_localize(None), lab["EndTime"].tz_localize(None)
            anomaly_mask |= (df_all.index >= start) & (df_all.index <= end)
            
        self.segments = []
        seg_id = 0
        
        # 1. Chop Anomalies with MIL
        anom_indices = np.where(anomaly_mask)[0]
        if len(anom_indices) > 0:
            # Mapping for physical event IDs to integers
            unique_ids = labels_df["ID"].unique().tolist()
            id_map = {id_str: i for i, id_str in enumerate(unique_ids)}
            
            blocks = np.split(anom_indices, np.where(np.diff(anom_indices) != 1)[0] + 1)
            for blk in blocks:
                event_pts = len(blk)
                bag_ts = []

                # Determine which physical ID this block belongs to
                mid_time = df_all.index[blk[len(blk)//2]]
                matching_ids = labels_df[(labels_df["StartTime"] <= mid_time) & (labels_df["EndTime"] >= mid_time)]["ID"]
                physical_id = matching_ids.iloc[0] if not matching_ids.empty else unique_ids[0]
                integer_id = id_map[physical_id]

                if event_pts <= actual_pts:
                    # Centered Padding
                    center_idx = blk[0] + len(blk) // 2
                    s_idx = max(0, center_idx - actual_pts // 2)
                    e_idx = min(len(df_all), s_idx + actual_pts)
                    bag_ts.append(df_all.iloc[s_idx:e_idx].values)
                else:
                    # Randomized MIL Splitting
                    n_windows = int(np.ceil(event_pts / actual_pts))
                    slack = n_windows * actual_pts - event_pts
                    offset = self.rng.integers(0, slack + 1)
                    curr_s = max(0, blk[0] - offset)
                    for _ in range(n_windows):
                        curr_e = curr_s + actual_pts
                        bag_ts.append(df_all.iloc[curr_s:curr_e].values)
                        curr_s = curr_e - (slack // (n_windows - 1) if n_windows > 1 else 0)

                valid_bag = [ts for ts in bag_ts if abs(len(ts) - actual_pts) < 5]
                if valid_bag:
                    self.segments.append({
                        "segment": seg_id, "ts": valid_bag, "label": 1, 
                        "event_id": integer_id
                    })
                    seg_id += 1

        # 2. Chop Nominals
        nom_indices = np.where(~anomaly_mask)[0]
        if len(nom_indices) > 0:
            blocks = np.split(nom_indices, np.where(np.diff(nom_indices) != 1)[0] + 1)
            for blk in blocks:
                for i in range(0, len(blk) - actual_pts, actual_pts):
                    ts_segment = df_all.iloc[blk[i : i + actual_pts]].values
                    self.segments.append({
                        "segment": seg_id, "ts": [ts_segment], "label": 0,
                        "event_id": -(seg_id + 1)
                    })
                    seg_id += 1