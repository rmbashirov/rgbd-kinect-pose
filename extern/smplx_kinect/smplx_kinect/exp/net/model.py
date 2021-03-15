import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hiddens, activation=nn.ReLU, xavier_init=True):
        super().__init__()
        layers = []
        for i in range(len(hiddens) - 1):
            layers.append(nn.Linear(hiddens[i], hiddens[i + 1]))
            if xavier_init:
                nn.init.xavier_uniform_(layers[-1].weight, gain=0.01)
            if i != len(hiddens) - 2:
                layers.append(activation())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PosePredictor(nn.Module):
    def __init__(self, hiddens, heads, xavier_init=True):
        super().__init__()
        self.mlp = MLP(hiddens, xavier_init=xavier_init)
        self.heads = heads
        for head_name, head_len in self.heads.items():
            setattr(self, head_name, nn.Linear(hiddens[-1], head_len))
            if xavier_init:
                nn.init.xavier_uniform_(getattr(self, head_name).weight, gain=0.01)

    def forward(self, x):
        x = self.mlp(x)
        result = {
            head_name: getattr(self, head_name)(x)
            for head_name, head_len in self.heads.items()
        }
        return result

    
class RNNPosePredictor(nn.Module):
    def __init__(self, input_size, heads, hidden_size=1000, num_layers=2, mlp_layers=None, xavier_init=True):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.h0 = nn.Parameter(torch.zeros(self.rnn.num_layers, 1, hidden_size).normal_(std=0.01), requires_grad=True)
        if mlp_layers is not None:
            self.mlp = MLP([hidden_size] + mlp_layers)
        else:
            self.mlp = None
        self.heads = dict()
        for head_name, head_len in heads.items():
            layer = nn.Linear(hidden_size if self.mlp is None else mlp_layers[-1], head_len)
            if xavier_init:
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
            self.heads[head_name] = layer

        self.heads = nn.ModuleDict(self.heads)

    def forward(self, x, h=None, return_all=False, return_x=False):
        if h is None:
            h = self.h0.expand(-1, x.shape[0], -1).contiguous()
        x, h = self.rnn(x, h)

        if not return_all:
            x = x[:, -1:]

        if self.mlp is not None:
            x = self.mlp(x)

        result = {
            head_name: module(x)
            for head_name, module in self.heads.items()
        }
        if return_x:
            return result, h, x
        else:
            return result, h
