from torch import nn


class FFNN(nn.Module):
    def __init__(self, size_vocabulary):
        super(FFNN, self).__init__()
        size_hidden_layer = int((size_vocabulary+10)/2)
        self.layers = nn.Sequential(
            nn.Linear(size_vocabulary, size_hidden_layer), 
            nn.ReLU(), 
            nn.Linear(size_hidden_layer, 10),
        )  

    def forward(self, x):
            output = self.layers(x)
            softmaxer = nn.Softmax()
            return softmaxer(output)
    

if __name__ == "__main__":
    test = FFNN()
    print(test)