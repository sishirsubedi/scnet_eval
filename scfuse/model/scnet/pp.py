import os
import pandas as pd
import numpy as np
import scanpy as sc
import torch
import networkx as nx
import torch
from torch_geometric.utils import  convert
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from .KNNDataset import KNNDataset, CellDataset
from torch.utils.data import DataLoader
import warnings


NETWORK_CUTOFF = 0.5
EXPRESSION_CUTOFF = 0.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')

def build_network(obj, net, biogrid_flag = False, human_flag = False):
    """
    Build a gene-gene network from the provided interaction information.
    Args:
      obj (anndata.AnnData): Single-cell data object (AnnData) containing gene expression data.
      net (pandas.DataFrame): DataFrame containing gene interactions (Source, Target, and Conn columns).
      biogrid_flag (bool, optional): If True, columns for net are set to ["Source", "Target"] only.
      human_flag (bool, optional): If True, keeps gene names unchanged; otherwise adjusts gene name casing.
    Returns:
      tuple:
        pandas.DataFrame: Filtered interaction DataFrame for valid genes.
        networkx.Graph: Graph representation of the gene network.
        pandas.DataFrame: Node-level gene expression features.
    """
    if not biogrid_flag:
        net.columns = ["Source","Target","Conn"]
        net = net.loc[net.Conn >= NETWORK_CUTOFF]
    
    else:
         net.columns = ["Source","Target"]
    
    if not human_flag:
        net["Source"] = net["Source"].apply(lambda x: x[0] + x[1:].lower()).astype(str)
        net["Target"] = net["Target"].apply(lambda x: x[0] + x[1:].lower()).astype(str)

         
    genes = list(pd.concat([net.Source, net.Target]).drop_duplicates())
    genes =  obj.var[obj.var.index.isin(genes)].index
    node_feature = sc.get.obs_df(obj.raw.to_adata(),list(genes)).T
    node_feature["non_zero"] = node_feature.apply(lambda x: x.astype(bool).sum(), axis=1)
    node_feature = node_feature.loc[node_feature.non_zero > node_feature.shape[1] * EXPRESSION_CUTOFF]
    node_feature.drop("non_zero",axis=1,inplace=True)

    net = net.loc[net.Source != net.Target]
    net = net.loc[net.Source.isin(node_feature.index)]
    net = net.loc[net.Target.isin(node_feature.index)]

    gp = nx.from_pandas_edgelist(net, "Source", "Target")

    node_feature = node_feature.loc[list(gp.nodes)]


    return net, gp, node_feature

def test_recon(model,x, data, knn_edge_index):
    """
    Evaluate model reconstruction performance on test edges.
    Args:
      model (torch.nn.Module): Trained scNET model.
      x (torch.Tensor): Input features for the nodes.
      data (torch_geometric.data.Data): Graph data object containing positive and negative edges.
      knn_edge_index (torch.Tensor): k-NN graph edges for the rows.
    Returns:
      float: AUC score of edge reconstruction.
    """
    model.eval()
    with torch.no_grad():
        embbed_rows, _, _ = model(x, knn_edge_index, data.train_pos_edge_index)
    return model.test(embbed_rows, data.test_pos_edge_index, data.test_neg_edge_index)

def pre_processing(adata,n_neighbors): 
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
   
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=15)

    return adata

def crate_knn_batch(knn,idxs,k=15):
  """
  Create a mini-batch of the k-NN graph for the given subset of indices.
  Args:
    knn (scipy.sparse.csr_matrix): Sparse adjacency matrix representing the k-NN graph.
    idxs (list[int]): List of indices used to subset the k-NN graph.
    k (int, optional): Number of nearest neighbors (used for reference if needed).
  Returns:
    torch.Tensor: Edge index for the sub-batch of the k-NN graph.
  """
  adjacency_matrix = torch.tensor(knn[idxs][:,idxs].toarray())
  row_indices, col_indices = torch.nonzero(adjacency_matrix, as_tuple=True)
  knn_edge_index = torch.stack((row_indices, col_indices))
  knn_edge_index = torch.unique(knn_edge_index, dim=1)
  return knn_edge_index.to(device)

def build_knn_graph(obj):
    graph = obj.obsp["distances"].toarray()
    graph = (graph > 0).astype(int)
    graph = nx.from_numpy_array(np.matrix(graph))
    ppi_geo = convert.from_networkx(graph)
    edge_index = ppi_geo.edge_index
    sc.pp.highly_variable_genes(obj)
    return edge_index, obj.var.highly_variable

def mini_batch_knn(edge_index, batch_size):
    """
    Create a mini-batch DataLoader for cells and their corresponding edges.
    Args:
      x (torch.Tensor): Matrix of gene expression features.
      edge_index (scipy.sparse.spmatrix): Distance or similarity matrix for cells.
      batch_size (int): Number of cells per mini-batch.
    Returns:
      torch.utils.data.DataLoader: DataLoader object for batching cells.
    Convert a NetworkX graph to a PyTorch Geometric edge index.
    Args:
      G (networkx.Graph): Input NetworkX graph.
      mapping (dict, optional): Dictionary mapping original node IDs to new indices.
    Returns:
      tuple:
        torch.Tensor: PyTorch Geometric edge index.
        dict: Mapping of graph nodes to tensor indices.
    """
    knn_dataset = KNNDataset(edge_index)
    knn_loader = DataLoader(knn_dataset,batch_size=batch_size, shuffle=True, drop_last=False)
    return knn_loader

def mini_batch_cells(x,edge_index, batch_size):
    cell_dataset = CellDataset(x, edge_index)
    cell_loader = DataLoader(cell_dataset,batch_size=batch_size, shuffle=False, drop_last=True)
    return cell_loader

def nx_to_pyg_edge_index(G, mapping=None):
    G = G.to_directed() if not nx.is_directed(G) else G
    if mapping is None:  
       mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long).to(device)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]
    return edge_index, mapping
