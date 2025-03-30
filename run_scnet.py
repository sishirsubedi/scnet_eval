
import gdown
download_url = f'https://drive.google.com/uc?id=1C_G14cWk95FaDXuXoRw-caY29DlR9CPi'
output_path = './data/example.h5ad'
gdown.download(download_url, output_path, quiet=False)


import scNET
import scanpy as sc

adata = sc.read_h5ad("./data/example.h5ad")


