import anndata as ad
import scanpy as sc 
import pandas as pd 

import scfuse

adata = sc.read_h5ad("./data/example.h5ad")

dfl = pd.read_parquet('scfuse_embed.pt')

dfl.index = adata.obs.index.values
adata.obsm['scfuse'] = dfl

adata.obs.leiden = None

import matplotlib.pylab as plt
sc.pp.neighbors(adata,use_rep='scfuse')
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata,color=['leiden','seurat_clusters'])
plt.tight_layout()
plt.savefig('scfuse_umap_example.png')