from torch import nn

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)

        self.relu = nn.ReLU()

    def forward(self, x):
        out =  self.layer2(self.relu(self.layer1(x)))
        return out