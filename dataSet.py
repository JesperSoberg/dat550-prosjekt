from torch.utils.data import Dataset

class CustomDataDataSet(Dataset):
	def __init__(self, tensor, labels):
		self.labels = labels
		self.abstracts = tensor 


	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		return self.abstracts[idx], self.labels[idx]