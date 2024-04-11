from torch import nn


class FFNN(nn.Module):
    def __init__(self, size_vocabulary):
        super(FFNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(size_vocabulary, 5), 
            nn.ReLU(), 
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 10),
        )  

    def forward(self, x):
            output = self.layers(x)
            return output
    

if __name__ == "__main__":
    test = FFNN()
    print(test)