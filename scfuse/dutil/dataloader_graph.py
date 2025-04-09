import torch
from torch.utils.data import Dataset

class KNNDataset(Dataset):
    def __init__(self, edge_index):
        self.edge_index = edge_index.T

    def __len__(self):
        return self.edge_index.shape[0]

    def __getitem__(self, idx):
        return self.edge_index[idx,:]
    
    