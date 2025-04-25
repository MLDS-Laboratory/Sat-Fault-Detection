
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class ESAMissionDataLoader:
    """
    Parameters
    ----------
    mission_dir : str
        Folder that contains channels.csv, labels.csv, anomaly_types.csv
        and the sub-folder "channels/" with pickled time-series files.

    nominal_segment_len : int or None, default None
        Maximum number of points for a *generated* nominal segment.
        If None, it uses the median length of all anomaly segments
        for that channel.

    train_ratio : float, default 0.8
        Fraction of segments that go to the training split
        (stratified on the anomaly flag).

    random_state : int, default 42
        For reproducible shuffles / splits.
    """

    def __init__(
        self,
        mission_dir: str,
        nominal_segment_len: int | None = None,
        train_ratio: float = 0.8,
        random_state: int = 42,
    ):
        self.dir = os.path.abspath(mission_dir)
        self.nominal_segment_len = nominal_segment_len
        self.train_ratio = train_ratio
        self.random_state = random_state

        # internal
        self.channels_df: pd.DataFrame | None = None
        self.labels_df: pd.DataFrame | None = None
        self.segments: list[dict] | None = None

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------
    def get_train_test_segments(self):
        """Return train / test lists (same API as OpsSatDataLoader)."""
        if self.segments is None:
            self._build_segments()

        # simple stratified split on anomaly flag
        labels = [s["label"] for s in self.segments]
        idx_train, idx_test = train_test_split(
            np.arange(len(self.segments)),
            test_size=1.0 - self.train_ratio,
            stratify=labels,
            random_state=self.random_state,
        )

        for i in idx_train:
            self.segments[i]["train"] = 1
        for i in idx_test:
            self.segments[i]["train"] = 0

        train_data = [self.segments[i] for i in idx_train]
        test_data = [self.segments[i] for i in idx_test]
        return train_data, test_data

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _load_meta(self):
        cfile = os.path.join(self.dir, "channels.csv")
        lfile = os.path.join(self.dir, "labels.csv")

        self.channels_df = pd.read_csv(cfile)
        self.labels_df = pd.read_csv(lfile)

        # unify column names just in case
        self.labels_df.rename(
            columns={
                "Channel": "Channel",
                "StartTime": "StartTime",
                "EndTime": "EndTime",
            },
            inplace=True,
        )
        # to pandas Timestamps
        self.labels_df["StartTime"] = pd.to_datetime(self.labels_df["StartTime"])
        self.labels_df["EndTime"] = pd.to_datetime(self.labels_df["EndTime"])

    def _build_segments(self):
        """Main routine: iterate over each channel and chop data."""
        self._load_meta()
        segments = []
        seg_id = 0

        ch_dir = os.path.join(self.dir, "channels")
        for _, ch_row in self.channels_df.iterrows():
            ch_name = ch_row["Channel"]
            pkl_path = os.path.join(ch_dir, ch_name)

            if not os.path.isfile(pkl_path):
                print(f"[WARN] Pickle for channel '{ch_name}' not found → skip")
                continue

            ts_df: pd.DataFrame = pd.read_pickle(pkl_path)
            ts_df = ts_df.sort_index()  # ensure ascending

            idx = ts_df.index
            values = ts_df.iloc[:, 0].astype(np.float32).values
            sampling_sec = (
                idx.to_series().diff().dt.total_seconds().median()
            )  # median sampling interval

            # All anomaly windows for this channel
            ch_labels = self.labels_df[self.labels_df["Channel"] == ch_name]

            # ------------------------------------------------------------------
            # --> 1) create segments for each anomaly window
            # ------------------------------------------------------------------
            for _, lab in ch_labels.iterrows():
                start, end = lab["StartTime"], lab["EndTime"]
                mask = (idx >= start) & (idx <= end)
                if not mask.any():
                    continue  # window outside available data

                ts_segment = values[mask]

                segments.append(
                    {
                        "segment": seg_id,
                        "channel": ch_name,
                        "ts": ts_segment,
                        "label": 1,
                        "sampling": sampling_sec,
                        "train": None,  # to be filled later
                    }
                )
                seg_id += 1

            # ------------------------------------------------------------------
            # --> 2) build nominal segments (gaps that are NOT in anomaly spans)
            # ------------------------------------------------------------------
            # boolean mask True if timestamp in ANY anomaly window
            if len(ch_labels) > 0:
                mask_anom = np.zeros(len(idx), dtype=bool)
                for _, lab in ch_labels.iterrows():
                    mask_anom |= (idx >= lab["StartTime"]) & (idx <= lab["EndTime"])
            else:
                mask_anom = np.zeros(len(idx), dtype=bool)

            # invert mask to get nominal indices
            nominal_idx = np.where(~mask_anom)[0]

            if len(nominal_idx) == 0:
                continue  # no nominal samples

            # group consecutive nominal indices → contiguous nominal blocks
            blocks = np.split(nominal_idx, np.where(np.diff(nominal_idx) != 1)[0] + 1)

            # define target length for nominal chunks
            if self.nominal_segment_len is not None:
                target_len = self.nominal_segment_len
            else:
                # median of anomaly lengths (if no anomalies → 2048)
                anom_lengths = [
                    len(s["ts"]) for s in segments if s["channel"] == ch_name and s["label"] == 1
                ]
                target_len = int(np.median(anom_lengths)) if anom_lengths else 2048

            for blk in blocks:
                start_idx = 0
                while start_idx < len(blk):
                    end_idx = min(start_idx + target_len, len(blk))
                    seg_slice = blk[start_idx:end_idx]

                    ts_segment = values[seg_slice]
                    if len(ts_segment) == 0:
                        break

                    segments.append(
                        {
                            "segment": seg_id,
                            "channel": ch_name,
                            "ts": ts_segment,
                            "label": 0,
                            "sampling": sampling_sec,
                            "train": None,  # filled later
                        }
                    )
                    seg_id += 1
                    start_idx = end_idx

        self.segments = segments
