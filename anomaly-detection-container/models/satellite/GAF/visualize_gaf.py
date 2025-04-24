import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.gridspec import GridSpec
from gaf_transform import compute_gaf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from pipelines.ops_sat_dataloader import OpsSatDataLoader

def visualize_single_gaf(segment, image_size=224, save_path=None):
    """
    Visualize a single GAF image along with its original time series.
    
    Parameters:
        segment (dict): Segment dictionary containing time series data
        image_size (int): Size for GAF image display
        save_path (str): Optional path to save the visualization
    """
    ts = segment['ts']
    label = segment['label']
    label_str = "Anomaly" if label == 1 else "Normal"
    channel = segment['channel']
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 7))
    
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
    
    # Print segment details
    print(f"Segment ID: {segment['segment']}")
    print(f"Channel: {channel}, Label: {label_str}, Sampling: {segment['sampling']}")
    print(f"Time series length: {len(ts)}")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(f"{save_path}_raw.png")
        print(f"Images saved to {save_path}_raw.png and {save_path}_processed.png")
    
    return fig

def visualize_grid(segments, grid_size=(8, 8), start_idx=0, image_size=112, save_path=None):
    """
    Visualize multiple GAF images in a grid.
    
    Parameters:
        segments (list): List of segment dictionaries
        grid_size (tuple): Grid dimensions (rows, cols)
        start_idx (int): Starting index in the segments list
        image_size (int): Size for each GAF image
        save_path (str): Optional path to save the visualization
    """
    rows, cols = grid_size
    max_images = rows * cols
    
    # Create a larger figure for the grid
    plt.figure(figsize=(cols*2, rows*2))
    
    for i in range(max_images):
        idx = start_idx + i
        if idx >= len(segments):
            break
            
        segment = segments[idx]
        ts = segment['ts']
        label = segment['label']
        label_str = "A" if label == 1 else "N"  # A for anomaly, N for normal
        
        # Compute GAF
        gaf_img = compute_gaf(ts)
        
        # Plot in grid
        plt.subplot(rows, cols, i+1)
        plt.imshow(np.array(gaf_img))
        plt.title(f"#{segment['segment']}\n{label_str}", fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Grid saved to {save_path}")

def visualize_grid_with_timeseries(segments, grid_size=(4, 4), start_idx=0, save_path=None):
    """
    Visualize multiple GAF images with their time series in a grid.
    Each segment gets a row with both the time series and GAF image.
    
    Parameters:
        segments (list): List of segment dictionaries
        grid_size (tuple): Grid dimensions (rows, cols)
        start_idx (int): Starting index in the segments list
        save_path (str): Optional path to save the visualization
    """
    rows, cols = grid_size
    max_images = rows * cols
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(cols*4, rows*3))
    gs = GridSpec(rows, 2*cols, figure=fig)
    
    for i in range(min(max_images, len(segments) - start_idx)):
        idx = start_idx + i
        segment = segments[idx]
        ts = segment['ts']
        label = segment['label']
        label_str = "Anomaly" if label == 1 else "Normal"
        
        # Row and column indices
        row = i // cols
        col = i % cols
        
        # Time series plot
        ax1 = fig.add_subplot(gs[row, 2*col])
        ax1.plot(ts)
        ax1.set_title(f"Seg #{segment['segment']} ({label_str})", fontsize=9)
        if row == rows-1:
            ax1.set_xlabel("Time")
        if col == 0:
            ax1.set_ylabel("Value")
        
        # GAF image
        gaf_img = compute_gaf(ts)
        gaf_img = (gaf_img - gaf_img.min()) / (gaf_img.max() - gaf_img.min() + 1e-8)
        
        ax2 = fig.add_subplot(gs[row, 2*col+1])
        ax2.imshow(gaf_img, cmap='viridis')
        ax2.set_title(f"GAF Image", fontsize=9)
        ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Detail grid saved to {save_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Visualize GAF images from OPS-SAT dataset.')
    parser.add_argument('--dataset_csv', type=str, 
                      default="C:\\Users\\varun\\Documents\\UMD\\Research\\MLDS\\Sat-Fault-Detection\\anomaly-detection-container\\data\\OPS-SAT\\dataset.csv", 
                      help='Path to dataset.csv')
    parser.add_argument('--segment_csv', type=str, 
                      default="C:\\Users\\varun\\Documents\\UMD\\Research\\MLDS\\Sat-Fault-Detection\\anomaly-detection-container\\data\\OPS-SAT\\segments.csv", 
                      help='Path to segments.csv')
    parser.add_argument('--segment_idx', type=int, default=0, 
                      help='Index of segment to visualize (for single visualization)')
    parser.add_argument('--mode', type=str, choices=['single', 'grid', 'detailed_grid'], 
                      default='single', help='Visualization mode')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[8, 8], 
                      help='Grid dimensions as rows cols (e.g., 8 8 for 8x8 grid)')
    parser.add_argument('--start_idx', type=int, default=0, 
                      help='Starting index for grid visualization')
    parser.add_argument('--save', type=str, default=None, 
                      help='Path to save visualization')
    parser.add_argument('--show_anomalies_only', action='store_true', 
                      help='Only show anomalous segments in grid view')
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths if needed
    dataset_csv = args.dataset_csv
    segment_csv = args.segment_csv
    if not os.path.isabs(dataset_csv):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_csv = os.path.abspath(os.path.join(current_dir, dataset_csv))
    if not os.path.isabs(segment_csv):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        segment_csv = os.path.abspath(os.path.join(current_dir, segment_csv))
    
    # Print paths to help with debugging
    print(f"Loading dataset from: {dataset_csv}")
    print(f"Loading segments from: {segment_csv}")
    
    # Load data
    data_loader = OpsSatDataLoader(dataset_csv, segment_csv)
    all_segments = data_loader.get_segments()
    
    # Filter for anomalies if requested
    if args.show_anomalies_only and args.mode != 'single':
        all_segments = [s for s in all_segments if s['label'] == 1]
        print(f"Showing only anomalous segments ({len(all_segments)} total)")
    
    if args.mode == 'single':
        # Validate segment_idx
        if args.segment_idx < 0 or args.segment_idx >= len(all_segments):
            print(f"Error: segment_idx must be between 0 and {len(all_segments)-1}")
            exit(1)
        
        # Visualize single segment
        visualize_single_gaf(all_segments[args.segment_idx], save_path=args.save)
        
    elif args.mode == 'grid':
        # Visualize as grid
        visualize_grid(all_segments, grid_size=tuple(args.grid_size), 
                       start_idx=args.start_idx, save_path=args.save)
        
    elif args.mode == 'detailed_grid':
        # Visualize detailed grid with time series
        # Use smaller grid for detailed view
        rows, cols = min(args.grid_size[0], 5), min(args.grid_size[1], 5)
        visualize_grid_with_timeseries(all_segments, grid_size=(rows, cols), 
                                    start_idx=args.start_idx, save_path=args.save)
    
    plt.show()



# ----------- Example usage: ------------------------
# python .\models\satellite\GAF\visualize_gaf.py --mode grid
# python .\models\satellite\GAF\visualize_gaf.py --mode single
# python .\models\satellite\GAF\visualize_gaf.py --mode detailed_grid --grid_size 4 4 --start_idx 0 --save detailed_grid.png
# python .\models\satellite\GAF\visualize_gaf.py --mode grid --show_anomalies_only