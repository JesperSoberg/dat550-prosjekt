import torch
import torch.nn as nn


class RNN(nn.Module):
	def __init__(self, inputSize, hiddenSize, numLayers, rnnType="rnn"):
		super(RNN, self).__init__()
		self.hiddenSize = hiddenSize
		self.numLayers=numLayers
		self.rnn = nn.RNN(input_size=inputSize, hidden_size=hiddenSize, num_layers=numLayers, dropout=0.5)

		self.fc = nn.Linear(hiddenSize, 10)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		#print("Forwarding...")
	
		h0 = torch.zeros(self.numLayers,self.hiddenSize).to(x.device)

		out, _ = self.rnn(x, h0)

		# Hent ut siste hidden state output
		#out = out[:, -1]

		# Hent ut siste hidden state output til fult koblet lag
		out = self.fc(out)
  
		out = torch.sigmoid(out)
		out = self.softmax(out)

		
		return out