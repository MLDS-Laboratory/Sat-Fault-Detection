import pandas as pd
import numpy as np


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
        if self.meta_df is None or self.seg_df is None:
            self.load_data()
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