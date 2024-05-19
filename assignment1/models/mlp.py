from torch import nn

def init_weight(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_normal_(m.weight)

class SimpleMLP(nn.Module):
    """
    """
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.stack = nn.Sequential(
            nn.Dropout1d(p = 0.2),
            nn.Linear(in_features=input_dim, out_features=12),
            nn.ReLU(),
            nn.Linear(in_features=12, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=output_dim)
        )
        self.stack.apply(init_weight)


    def forward(self, inputs):
        """
        """
        return self.stack.forward(inputs)
