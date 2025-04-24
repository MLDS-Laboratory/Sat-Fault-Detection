"""
validate_mission1.py

Validate and preview ESA Mission channel files that are pickled pandas objects.
"""

import sys
import struct
import pandas as pd
from pathlib import Path

MAGIC = b"\x80\x05"  # protocol-5; change to b"\x80\x04" if you expect protocol-4 as well


def main(path: Path, n_rows: int = 5):
    # --- quick header check -------------------------------------------------
    with path.open("rb") as fh:
        hdr = fh.read(2)
    if hdr != MAGIC and hdr != b"\x80\x04":
        raise ValueError(f"{path.name} doesn’t look like a pickle "
                         f"(starts with {hdr.hex()}).")

    # --- load with pandas ---------------------------------------------------
    try:
        obj = pd.read_pickle(path)
    except Exception as e:
        raise RuntimeError(f"pandas.read_pickle failed: {e}")

    # --- show basic info ----------------------------------------------------
    print(f"\nLoaded object type : {type(obj)}")
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        print(f"Shape             : {obj.shape}")
        print(f"Index dtype       : {obj.index.dtype}")
        print(f"Head ({n_rows} rows):")
        print(obj.head(n_rows))
    else:
        print("First few entries:\n", obj)

    # extra sanity—timestamps should be monotonic
    if hasattr(obj.index, "is_monotonic_increasing"):
        print("\nIndex monotonic increasing:", obj.index.is_monotonic_increasing)


    # find where there are time gaps greater than x minutes in the data and display the timestamps and indices
    # Assuming the index is a DatetimeIndex
    if not isinstance(obj.index, pd.DatetimeIndex):
        # Try converting to DatetimeIndex if possible
        try:
            obj.index = pd.to_datetime(obj.index)
        except Exception as e:
            print(f"Unable to convert index to datetime: {e}")
            return

    gap_threshold = 20
    print(f"\nTime gaps greater than {gap_threshold} minutes:")
    for i in range(1, len(obj.index)):
        gap = obj.index[i] - obj.index[i - 1]
        if gap > pd.Timedelta(minutes=gap_threshold):
            print(f"Gap from index {i-1} ({obj.index[i-1]}) to index {i} ({obj.index[i]}): Gap = {gap}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python validate_mission1.py <path/to/channel_1> [rows]")
    file_path = Path(sys.argv[1]).expanduser()
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(file_path, rows)
