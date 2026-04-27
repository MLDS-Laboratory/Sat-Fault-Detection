import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gaf_transform import compute_gaf

def visualize_mission2_comparison_grid(mission_dir, channel_name="channel_18", save_path=None):
    """
    Creates a 2x3 grid comparing Nominal, True Anomaly, and Rare Nominal Event (RNE).
    Uses specific examples from Mission 2 referenced in the ESA-ADB paper.
    """
    pkl_path = os.path.join(mission_dir, "channels", channel_name)
    
    if not os.path.exists(pkl_path):
        print(f"Error: Could not find data at {pkl_path}")
        return
        
    print(f"Loading {channel_name} from {mission_dir}...")
    ts_df = pd.read_pickle(pkl_path).sort_index()
    
    # 1. True Anomaly: ID 631 (Start: 2001-12-14 19:16:29, Duration: ~1h 18m)
    anom_start = pd.to_datetime("2001-12-14 19:16:29")
    anom_end = anom_start + pd.Timedelta(hours=1, minutes=18)
    
    # 2. Rare Nominal Event (RNE): ID 591 (Start: 2002-04-16 16:30:53, Duration: ~35m)
    rne_start = pd.to_datetime("2002-04-16 16:30:53")
    rne_end = rne_start + pd.Timedelta(minutes=35)
    
    # 3. Nominal Event: A clean 1-hour window prior to the anomaly
    nom_start = pd.to_datetime("2001-12-13 12:00:00")
    nom_end = nom_start + pd.Timedelta(hours=6)
    
    # Check for and apply timezone localization if the pickle file uses it
    if ts_df.index.tz is not None:
        anom_start = anom_start.tz_localize(ts_df.index.tz)
        anom_end = anom_end.tz_localize(ts_df.index.tz)
        rne_start = rne_start.tz_localize(ts_df.index.tz)
        rne_end = rne_end.tz_localize(ts_df.index.tz)
        nom_start = nom_start.tz_localize(ts_df.index.tz)
        nom_end = nom_end.tz_localize(ts_df.index.tz)

    # Extract the raw values into numpy arrays
    nom_ts = ts_df.loc[nom_start:nom_end].iloc[:, 0].values
    anom_ts = ts_df.loc[anom_start:anom_end].iloc[:, 0].values
    rne_ts = ts_df.loc[rne_start:rne_end].iloc[:, 0].values

    segments = [
        {"name": "Nominal Telemetry", "ts": nom_ts},
        {"name": "True Anomaly\n(Event ID 631)", "ts": anom_ts},
        {"name": "Rare Nominal Event\n(Event ID 591)", "ts": rne_ts}
    ]

    # Plotting Setup: 2 Rows, 3 Columns
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    
    for col_idx, seg in enumerate(segments):
        ts = seg["ts"]
        title = seg["name"]
        
        if len(ts) == 0:
            axes[0, col_idx].set_title(f"{title} - NO DATA FOUND", fontsize=16, fontweight='bold')
            continue
            
        # --- ROW 0: Original Time Series ---
        ax_ts = axes[0, col_idx]
        ax_ts.plot(ts, color='#283593', linewidth=1.5)
        ax_ts.set_title(f"{title}", fontsize=18, fontweight='bold', pad=15)
        ax_ts.set_xlabel("Time Step", fontsize=16, fontweight='bold')
        ax_ts.set_ylabel("Sensor Value", fontsize=16, fontweight='bold')
        ax_ts.tick_params(axis='both', labelsize=13)
        ax_ts.grid(True, linestyle='--', alpha=0.6)
        
        # --- ROW 1: GAF Image ---
        ax_gaf = axes[1, col_idx]
        gaf_img = compute_gaf(ts)
        
        # Normalize the GAF image for clear visualization
        gaf_img = (gaf_img - gaf_img.min()) / (gaf_img.max() - gaf_img.min() + 1e-8)
        
        # aspect='auto' matches the GAF width to the time series plot above it
        im = ax_gaf.imshow(gaf_img, cmap='viridis', aspect='auto')
        
        ax_gaf.set_title(f"GAF Spatial Encoding", fontsize=18, fontweight='bold', pad=15)
        ax_gaf.set_xlabel("Time Step", fontsize=16, fontweight='bold')
        ax_gaf.set_ylabel("Time Step", fontsize=16, fontweight='bold')
        ax_gaf.tick_params(axis='both', labelsize=13)
        
        # Format Colorbar
        cbar = fig.colorbar(im, ax=ax_gaf, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=13)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison grid successfully saved to {save_path}")
        
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize Nominal vs Anomaly vs RNE for Mission 2')
    parser.add_argument('--mission_dir', type=str, 
                        help='Path to the ESA-ADB Mission2 directory',
                        default=os.path.abspath(os.path.join(__file__, "../../../../../data/ESA-Anomaly/ESA-Mission2")))
    parser.add_argument('--save', type=str, default='mission2_class_comparison.png', 
                        help='Path to save the resulting grid image')
    args = parser.parse_args()
    
    visualize_mission2_comparison_grid(args.mission_dir, save_path=args.save)