import torch
import torch.nn as nn

import random


seed = 42

class RNN(nn.Module):
	def __init__(self, inputSize, hiddenSize, numLayers, rnnType="rnn"):
		super(RNN, self).__init__()
		self.hiddenSize = hiddenSize
		self.numLayers=numLayers
		self.rnn = nn.RNN(input_size=inputSize, hidden_size=hiddenSize, num_layers=numLayers, batch_first=True)

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
			

def train(trainModel, optimizer="adam", model_type="RNN", x=None, labels=None):
	print("Training...")
	data = x
	criterion = nn.CrossEntropyLoss()#BCELoss()
	learnRate = 0.01

	optimizer = torch.optim.Adam(trainModel.parameters(), lr=learnRate)

	numEpochs = 100
	for epoch in range(numEpochs):
		prediction = trainModel(data)
		loss = criterion(prediction, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if epoch % 10 == 0:  # Print every 10 epochs
			print(f'Epoch {epoch}, Loss: {loss.item()}')

	return trainModel
