import json
import torch
import numpy as np
from tqdm import tqdm

def preprocess_vol(data_path, bins, window):
    with open(data_path, 'r') as f:
        all_implied_vol_list = json.load(f)
        
    for idx in range(len(all_implied_vol_list)):
        mean_vol = np.nanmean(all_implied_vol_list[idx])
        for sub_idx in range(bins):
            if np.isnan(all_implied_vol_list[idx][sub_idx]):
                all_implied_vol_list[idx][sub_idx] = mean_vol
    
    data = []
    for idx in range(window+1, len(all_implied_vol_list)):
        data.append(torch.tensor(all_implied_vol_list[idx-window-1:idx], dtype=torch.float32))
    
    return data