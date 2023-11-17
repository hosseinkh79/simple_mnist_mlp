#model
from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, in_features, num_classes, list_hidden_features_size):
        super().__init__()
        # self.num_hidden_layer = num_hidden_layer

        self.input_layer = nn.Linear(in_features, list_hidden_features_size[0])
        
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(list_hidden_features_size[i], list_hidden_features_size[i+1]) for i in range(len(list_hidden_features_size)-1)]
        )

        self.out_layer = nn.Linear(list_hidden_features_size[-1], num_classes)

    def forward(self, x):
        x = torch.relu(self.input_layer(x)) 
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))

        x = self.out_layer(x)
        
        return x