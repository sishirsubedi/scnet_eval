import anndata as ad
import scanpy as sc 
import pandas as pd 

import scfuse

adata = sc.read_h5ad("./data/example.h5ad")


adata = adata[:,adata.var.highly_variable]

ppi_net = pd.read_csv("./resources/format_h_sapiens.csv")[["g1_symbol","g2_symbol","conn"]].drop_duplicates()


human_flag = False
net, ppi, node_feature = scfuse.model.scnet.build_network(adata, ppi_net,human_flag=human_flag)



ppi_edge_index, _ = scfuse.model.scnet.nx_to_pyg_edge_index(ppi)
ppi_edge_index = ppi_edge_index


import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import  convert

adata = adata[:,node_feature.index]
cell_graph = adata.obsp["distances"].toarray()
cell_graph = (cell_graph > 0).astype(int)
cell_graph = nx.from_numpy_array(np.matrix(cell_graph))
cci_geo = convert.from_networkx(cell_graph)
knn_edge_index = cci_geo.edge_index



##gene expression data for genes as nodes
node_feature.shape
### gene-gene network
ppi_edge_index.shape
ppi_edge_index[0,:].max()
### cell-cell network
knn_edge_index.shape
knn_edge_index[0,:].max()


row_x = node_feature.T.values
row_x_label = np.array(range(node_feature.shape[1]))
row_x = torch.tensor(row_x, dtype=torch.float32).cpu()



INTER_DIM = 250
EMBEDDING_DIM = 75
NETWORK_CUTOFF = 0.5
MAX_CELLS_BATCH_SIZE = 4000
MAX_CELLS_FOR_SPLITING = 10000
DE_GENES_NUM = 3000
EXPRESSION_CUTOFF = 0.0
NUM_LAYERS = 3
EPS = 1e-15
batch_size = 100
device = 'cuda'
max_epoch = 100

def update_batch(batch, c_x_indx_map):
    for i in range(batch.size(0)):
        for j in range(batch.size(1)):
            if batch[i, j].item() in c_x_indx_map:
                batch[i, j] = c_x_indx_map[batch[i, j].item()]
    return batch



import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

data = Data(row_x,ppi_edge_index)
data = data.to(device)

knn_dataset = scfuse.dataloader_graph.KNNDataset(knn_edge_index)
knn_loader = DataLoader(knn_dataset,batch_size=batch_size, shuffle=True, drop_last=False)



model = scfuse.model.scnet.scNET( data.x.shape[1],batch_size *2,
INTER_DIM, EMBEDDING_DIM, INTER_DIM, EMBEDDING_DIM, lambda_rows = 1, lambda_cols=1,num_layers=NUM_LAYERS).to(device)


### train 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

for epoch in range(max_epoch):

        total_row_loss = 0
        total_col_loss = 0

        for _,batch in enumerate(knn_loader):
            
            c_x_indxs =  batch.flatten().unique()
            
            if len(c_x_indxs) != batch_size *2:
                continue
            
            c_x = data.x[c_x_indxs,:].clone()
            c_x = c_x.T
                        
            c_x_indx_map = {int(y):x for x,y in enumerate(c_x_indxs)}

            updated_batch = update_batch(batch, c_x_indx_map)

            row_embed, col_embed, out_features = model(c_x.to(device), updated_batch.T.to(device), data.edge_index)


            row_loss = model.recon_loss(row_embed, data.edge_index,sig=True)
            
            
            out_features = (out_features - (out_features.mean(axis=0)))/ (out_features.std(axis=0)+ EPS)
            reg = model.recon_loss(out_features.T, data.edge_index, sig=False)
            col_loss = model.feature_critarion(c_x.T, out_features)
            
            loss =  row_loss + col_loss + reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_row_loss += row_loss.detach().cpu().numpy()
            total_col_loss += (col_loss.detach().cpu().numpy()+reg.detach().cpu().numpy())
                        
        print('epoch '+ str(epoch)+ '  rl...'+ str(total_row_loss) + '  cl...'+ str(total_col_loss))


torch.save(model.state_dict(),'scfuse_example_model.pt')



### infer

model.load_state_dict(torch.load('scfuse_example_model.pt', weights_only=True, map_location=torch.device(device)))

model.eval()

dfl = pd.DataFrame()
for _,batch in enumerate(knn_loader):
    
    c_x_indxs =  batch.flatten()
    c_x = data.x[c_x_indxs,:].clone()
    c_x = c_x.T
    
    if c_x.shape[1] != batch_size * 2:
        continue
                
    c_x_indx_map = {int(y):x for x,y in enumerate(c_x_indxs)}

    c_x_indxs_before = c_x_indxs.detach().cpu().numpy().copy()
    updated_batch = update_batch(batch, c_x_indx_map)

    row_embed, col_embed, out_features = model(c_x.to(device), updated_batch.T.to(device), data.edge_index)
    
    df = pd.DataFrame(col_embed.detach().cpu().numpy())
    df.index = c_x_indxs_before
   
    dfl = pd.concat([dfl, df[~df.index.isin(dfl.index)]])

dfl = dfl[~dfl.index.duplicated(keep='last')]
dfl = dfl.sort_index()
dfl.to_parquet('scfuse_embed.pt',compression='gzip')