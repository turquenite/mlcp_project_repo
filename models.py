import torch
from swish import Swish

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.activation_function = torch.nn.ELU()
        self.last_layer_acivation_function = torch.nn.Sigmoid()

        self.linear_1 = torch.nn.Linear(7700, 255)
        self.linear_2 = torch.nn.Linear(255, 255)
        self.linear_3 = torch.nn.Linear(255, 64)
        self.linear_4 = torch.nn.Linear(64, 255)
        self.linear_5 = torch.nn.Linear(255, 128)
        self.linear_6 = torch.nn.Linear(128, 64)
        self.linear_7 = torch.nn.Linear(64, 32)
        self.linear_8 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation_function(x)
        x = self.linear_2(x)
        x = self.activation_function(x)
        x = self.linear_3(x)
        x = self.activation_function(x)
        x = self.linear_4(x)
        x = self.activation_function(x)
        x = self.linear_5(x)
        x = self.activation_function(x)
        x = self.linear_6(x)
        x = self.activation_function(x)
        x = self.linear_7(x)
        x = self.activation_function(x)
        x = self.linear_8(x)
        x = self.last_layer_acivation_function(x)

        x = torch.squeeze(x)

        return x
    

class DeepModel(torch.nn.Module):
    def __init__(self):
        super(DeepModel, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.BatchNorm1d(7700),
            torch.nn.Linear(7700, 10000),
            Swish(),
            torch.nn.BatchNorm1d(10000),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(10000, 5000),
            Swish(),
            torch.nn.BatchNorm1d(5000),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(5000, 2500),
            Swish(),
            torch.nn.BatchNorm1d(2500),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(2500, 1000),
            Swish(),
            torch.nn.BatchNorm1d(1000),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(1000, 64),
            Swish(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

        # Initialize weights using Xavier initialization
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)


    def forward(self, x):
        x = self.layers(x)

        return torch.squeeze(x)
