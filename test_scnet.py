
# import gdown

# import scNET.Models
# import scNET.main
# download_url = f'https://drive.google.com/uc?id=1C_G14cWk95FaDXuXoRw-caY29DlR9CPi'
# output_path = './data/example.h5ad'
# gdown.download(download_url, output_path, quiet=False)


import scNET
import scanpy as sc


adata = sc.read_h5ad("./data/example.h5ad")




pre_processing_flag = True 
biogrid_flag = False
human_flag=False
number_of_batches=5
split_cells = False
n_neighbors=25
max_epoch=150
model_name=""
save_model_flag = False

import numpy as np
import pandas as pd
import torch
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# adata = scNET.main.pre_processing(adata,n_neighbors)

'''
here, obj contains gene expression data of cell x gene
cell = 9172 and gene = 18531

'''


net = pd.read_csv("scNET/Data/format_h_sapiens.csv")[["g1_symbol","g2_symbol","conn"]].drop_duplicates()

'''
net data frame with three columns gene1,gene2,and conn score,
df has ~500k edges with
connection values 0.1 to 1.0
'''

net, ppi, node_feature = scNET.main.build_network(adata, net,human_flag=human_flag)

'''
net --> build_network trims net dataframe such that now conn is only 87k from 500k, scores from 0.6 to 1.0


ppi --> this is graph where edges are 87k gene-gene edges and nodes are nodes are 11049 genes


so out of 18531 genes in adata we have 11049 nodes as genes


node_feature--> this is data frame with gene 11049 x 9172 cells 

'''

'''
so what do we have--

adata is anndata with 9k cells and 18k genes
net is df with 87k rows of gene-gene-score
ppi is graph of 11k genes and 87k edges
node_feature is df with 11k genes and 9k cells

'''

device='cuda'
ppi_edge_index, _ = scNET.main.nx_to_pyg_edge_index(ppi)
ppi_edge_index = ppi_edge_index.to(device)


'''
convert ppi graph to tensor with 2 rows where row 1 is index of 11k genes and row 2 is its interacting node gene

'''


'''select matching gene names from main expression obj anndata
now, obj has matching 9k cells and 11k genes

'''
adata = adata[:,node_feature.index]


'''
now for gene expression data-
calculate cell to cell distance
use nx to convert to graph with cells as node
then convert to tensor edge index 
'''
# sc.pp.highly_variable_genes(obj)
highly_variable_index =  adata.var.highly_variable 
from scNET.main import nx, convert
graph = adata.obsp["distances"].toarray()
graph = (graph > 0).astype(int)
graph = nx.from_numpy_array(np.matrix(graph))
ppi_geo = convert.from_networkx(graph)
knn_edge_index = ppi_geo.edge_index


##gene expression data for genes as nodes
node_feature.shape
### gene-gene network
ppi_edge_index.shape
ppi_edge_index[0,:].max()
### cell-cell network
knn_edge_index.shape
knn_edge_index[0,:].max()



#### for gene - load all exp and ppi
x = node_feature.values
x = torch.tensor(x, dtype=torch.float32).cpu()
data = scNET.main.Data(x,ppi_edge_index)



batch_size=64
### for cells - create dataloader
knn_dataset = scNET.main.KNNDataset(knn_edge_index)
knn_loader = torch.utils.data.DataLoader(knn_dataset,batch_size=batch_size, shuffle=True, drop_last=False)


INTER_DIM = 250
EMBEDDING_DIM = 75
NETWORK_CUTOFF = 0.5
MAX_CELLS_BATCH_SIZE = 4000
MAX_CELLS_FOR_SPLITING = 10000
DE_GENES_NUM = 3000
EXPRESSION_CUTOFF = 0.0
NUM_LAYERS = 3

x_full = data.x.clone()
model = scNET.main.scNET(x_full.shape[0], x_full.shape[1], INTER_DIM, EMBEDDING_DIM, INTER_DIM, EMBEDDING_DIM, lambda_rows = 1, lambda_cols=1, num_layers=NUM_LAYERS).to(device)

