import logging



class scfuse(object):
	"""
	Initialize scfuse model

	Parameters
	----------
	data: 
	wdir: path to save model outputs            
	sample: name of data sample            
				
	Returns
	-------
	None

	"""
	def __init__(self, 
		data: list, 
		sample: str,
		wdir: str
		):
	 
	 
		self.data = data
		self.sample = sample
		self.wdir = wdir
				
		print(self.wdir)
		logging.basicConfig(filename=self.wdir+'scfuse_model.log',
		format='%(asctime)s %(levelname)-8s %(message)s',
		level=logging.INFO,
		datefmt='%Y-%m-%d %H:%M:%S')
  		 		
def create_scfuse_object(
	data_list: list,
	sample:str, 
	wdir:str
	):
	return scfuse(data_list,sample,wdir)
