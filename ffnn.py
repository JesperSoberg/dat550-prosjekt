from torch import nn


class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3606, 5), 
            nn.ReLU(), 
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )  

    def forward(self, x):
            output = self.layers(x)
            return output
    

if __name__ == "__main__":
    test = FFNN()
    print(test)