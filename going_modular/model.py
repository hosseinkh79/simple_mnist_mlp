#model
from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, in_features, num_classes, list_hidden_features_size):
        super().__init__()

        self.input_layer = nn.Linear(in_features, list_hidden_features_size[0])
        self.batch_norm_input = nn.BatchNorm1d(list_hidden_features_size[0])

        self.hidden_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(list_hidden_features_size[i], list_hidden_features_size[i + 1]),
                nn.BatchNorm1d(list_hidden_features_size[i + 1]),
                nn.ReLU(inplace=True)
            ) for i in range(len(list_hidden_features_size) - 1)]
        )

        self.out_layer = nn.Linear(list_hidden_features_size[-1], num_classes)

    def __repr__(self):
        return(f'Model:{type(self).__name__} (num_hyden_layers:{len(self.hidden_layers)})')

    def forward(self, x):
        x = self.batch_norm_input(torch.relu(self.input_layer(x)))

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.out_layer(x)

        return x