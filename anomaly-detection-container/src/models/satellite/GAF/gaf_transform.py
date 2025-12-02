import numpy as np

def normalize_series(x):
    """
    Normalize a 1-D time series x into the range [-1, 1].
    """
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val - min_val == 0:
        return np.zeros_like(x)
    return 2 * (x - min_val) / (max_val - min_val) - 1

def compute_gaf(x):
    """
    Compute the Gramian Angular Field for a 1-D time series x.
    
      1. Normalize x to [-1, 1].
      2. Compute the angular representation: phi = arccos(x).
      3. Create the GAF matrix: G[i, j] = cos(phi_i + phi_j).
      
    The resulting 2-D matrix encodes pairwise temporal relationships.
    """
    x = np.array(x, dtype=np.float32)
    x_norm = normalize_series(x)
    phi = np.arccos(x_norm)  # phi is in [0, pi]
    
    # Compute GAF via broadcasting
    phi_i = phi.reshape(-1, 1)
    phi_j = phi.reshape(1, -1)
    gaf = np.cos(phi_i + phi_j)
    return gaf
