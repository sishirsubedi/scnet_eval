import anndata as ad
import scanpy as sc 
import pandas as pd 

import scfuse 

adata = sc.read_h5ad("./data/example.h5ad")

ppi_net = pd.read_csv("./resources/format_h_sapiens.csv")[["g1_symbol","g2_symbol","conn"]].drop_duplicates()

scfuse.model.scnet.