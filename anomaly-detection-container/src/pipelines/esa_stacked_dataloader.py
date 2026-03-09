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
        self.segments = None

    def get_train_test_segments(self):
        if self.segments is None:
            self._build_stacked_segments()

        labels = [s["label"] for s in self.segments]
        idx_train, idx_test = train_test_split(
            np.arange(len(self.segments)), test_size=1.0 - self.train_ratio,
            stratify=labels, random_state=self.random_state
        )
        return [self.segments[i] for i in idx_train], [self.segments[i] for i in idx_test]

    def _build_stacked_segments(self):
        cfile = os.path.join(self.dir, "channels.csv")
        lfile = os.path.join(self.dir, "labels.csv")
        channels_df = pd.read_csv(cfile)
        labels_df = pd.read_csv(lfile)
        
        labels_df["StartTime"] = pd.to_datetime(labels_df["StartTime"])
        labels_df["EndTime"] = pd.to_datetime(labels_df["EndTime"])

        ch_dir = os.path.join(self.dir, "channels")
        all_series = {}
        intervals = []

        # 1. Load all channels and find the median sampling frequency
        for _, ch_row in channels_df.iterrows():
            ch_name = ch_row["Channel"]
            pkl_path = os.path.join(ch_dir, ch_name)
            if not os.path.isfile(pkl_path): continue
            
            ts_df = pd.read_pickle(pkl_path).sort_index()
            # Remove timezone for clean merging
            ts_df.index = ts_df.index.tz_localize(None) 
            all_series[ch_name] = ts_df.iloc[:, 0]
            
            interval = ts_df.index.to_series().diff().dt.total_seconds().median()
            if pd.notna(interval): intervals.append(interval)

        global_freq = f"{int(np.median(intervals))}s" if intervals else "1s"
        
        # 2. Build the unified DataFrame via interpolation
        df_all = pd.DataFrame(all_series)
        df_all = df_all.resample(global_freq).interpolate(method='linear').dropna()
        
        # 3. Create global anomaly mask (1 if ANY channel is anomalous at that timestamp)
        anomaly_mask = np.zeros(len(df_all), dtype=bool)
        for _, lab in labels_df.iterrows():
            start, end = lab["StartTime"].tz_localize(None), lab["EndTime"].tz_localize(None)
            anomaly_mask |= (df_all.index >= start) & (df_all.index <= end)
            
        # 4. Chop into segments
        self.segments = []
        seg_id = 0
        
        # Chop Anomalies
        anom_indices = np.where(anomaly_mask)[0]
        if len(anom_indices) > 0:
            blocks = np.split(anom_indices, np.where(np.diff(anom_indices) != 1)[0] + 1)
            for blk in blocks:
                ts_segment = df_all.iloc[blk].values # Shape: (Seq_Len, Num_Channels)
                self.segments.append({"segment": seg_id, "ts": ts_segment, "label": 1})
                seg_id += 1

        # Chop Nominals
        nom_indices = np.where(~anomaly_mask)[0]
        if len(nom_indices) > 0:
            blocks = np.split(nom_indices, np.where(np.diff(nom_indices) != 1)[0] + 1)
            for blk in blocks:
                start_idx = 0
                while start_idx < len(blk):
                    end_idx = min(start_idx + self.nominal_segment_len, len(blk))
                    ts_segment = df_all.iloc[blk[start_idx:end_idx]].values
                    self.segments.append({"segment": seg_id, "ts": ts_segment, "label": 0})
                    seg_id += 1
                    start_idx = end_idx