from torch import nn


class FFNN(nn.Module):
    def __init__(self, size_vocabulary, num_hidden_layers=1):
        super(FFNN, self).__init__()
        size_hidden_layer = int((size_vocabulary+10)/2)    

        layers = [nn.Linear(size_vocabulary, size_hidden_layer), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(size_hidden_layer, size_hidden_layer))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(size_hidden_layer, 10))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
            output = self.layers(x)
            softmaxer = nn.Softmax(dim=0)
            return softmaxer(output)
    

if __name__ == "__main__":
    test = FFNN(size_vocabulary=3700, num_hidden_layers=5)
    print(test)