import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from constants import KAGGLE_DATA_PATH

class RNA3D_Dataset(Dataset):
    def __init__(self, indices, data, max_len=384):
        self.indices = indices
        self.data = data
        self.tokens = {nt:i for i,nt in enumerate('ACGU')}
        self.max_len = max_len

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        sequence = [self.tokens[nt] for nt in (self.data['sequence'][idx])]
        sequence = np.array(sequence)
        sequence = torch.tensor(sequence)

        # Get C1' xyz
        xyz = self.data['xyz'][idx]
        xyz = torch.tensor(np.array(xyz))

        if len(sequence) > self.max_len:
            crop_start = np.random.randint(len(sequence) - self.max_len)
            crop_end = crop_start + self.max_len

            sequence = sequence[crop_start:crop_end]
            xyz = xyz[crop_start:crop_end]

        # Return length for batch padding
        return {'sequence': sequence, 'xyz': xyz, 'length': len(sequence)}

def collate_batch(batch):
    """
    Custom collate function to handle padding in batches
    """
    # Get the maximum sequence length in this batch
    max_len = max([item['length'] for item in batch])
    
    # Initialize tensors
    batch_size = len(batch)
    sequences = torch.zeros((batch_size, max_len), dtype=torch.long)
    xyz = torch.zeros((batch_size, max_len, 3), dtype=torch.float)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    # Fill tensors with data
    for i, item in enumerate(batch):
        seq_len = item['length']
        sequences[i, :seq_len] = item['sequence']
        xyz[i, :seq_len] = item['xyz']
        mask[i, :seq_len] = True
    
    return {
        'sequence': sequences,
        'xyz': xyz,
        'mask': mask,
        'lengths': [item['length'] for item in batch]
    }

def load_and_preprocess_data(config):
    """Load and preprocess RNA sequence and structure data"""
    print("Loading and preprocessing data...")
    
    # Load data
    train_sequences = pd.read_csv(f"{KAGGLE_DATA_PATH}/train_sequences.csv")
    train_labels = pd.read_csv(f"{KAGGLE_DATA_PATH}/train_labels.csv")
    train_labels["pdb_id"] = train_labels["ID"].apply(lambda x: x.split("_")[0]+'_'+x.split("_")[1])
    
    # Extract coordinates
    all_xyz = []
    for pdb_id in tqdm(train_sequences['target_id']):
        df = train_labels[train_labels["pdb_id"] == pdb_id]
        xyz = df[['x_1','y_1','z_1']].to_numpy().astype('float32')
        xyz[xyz < -1e17] = float('Nan')
        all_xyz.append(xyz)
    
    # Filter data
    filter_nan = []
    max_len = 0
    for xyz in all_xyz:
        if len(xyz) > max_len:
            max_len = len(xyz)
        
        filter_nan.append((np.isnan(xyz).mean() <= 0.5) & 
                          (len(xyz) < config['max_len_filter']) & 
                          (len(xyz) > config['min_len_filter']))
    
    print(f"Longest sequence in train: {max_len}")
    
    filter_nan = np.array(filter_nan)
    non_nan_indices = np.arange(len(filter_nan))[filter_nan]
    
    train_sequences = train_sequences.loc[non_nan_indices].reset_index(drop=True)
    all_xyz = [all_xyz[i] for i in non_nan_indices]
    
    # Pack data into a dictionary
    data = {
        "sequence": train_sequences['sequence'].to_list(),
        "temporal_cutoff": train_sequences['temporal_cutoff'].to_list(),
        "description": train_sequences['description'].to_list(),
        "all_sequences": train_sequences['all_sequences'].to_list(),
        "xyz": all_xyz
    }
    
    return data

def create_train_val_split(data, cutoff_date, test_cutoff_date):
    """Split data into training and validation sets based on temporal cutoff"""
    cutoff_date_ts = pd.Timestamp(cutoff_date)
    test_cutoff_date_ts = pd.Timestamp(test_cutoff_date)
    
    train_index = [i for i, d in enumerate(data['temporal_cutoff']) 
                  if pd.Timestamp(d) <= cutoff_date_ts]
    
    test_index = [i for i, d in enumerate(data['temporal_cutoff']) 
                 if pd.Timestamp(d) > cutoff_date_ts and pd.Timestamp(d) <= test_cutoff_date_ts]
    
    print(f"Train size: {len(train_index)}")
    print(f"Test size: {len(test_index)}")
    
    return train_index, test_index

def get_data_loaders(config):
    """Create and return data loaders for training and validation"""
    data = load_and_preprocess_data(config)
    train_idx, val_idx = create_train_val_split(
        data, config['cutoff_date'], config['test_cutoff_date'])
    
    train_dataset = RNA3D_Dataset(train_idx, data, config['max_len'])
    val_dataset = RNA3D_Dataset(val_idx, data, config['max_len'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=collate_batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        collate_fn=collate_batch
    )
    
    return train_loader, val_loader 