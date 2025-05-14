import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

device = "cuda" if torch.cuda.is_available() else "cpu"


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim):
        super(NN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))    #first layer (maps the input to the hidden layer)
        for _ in range(hidden_num - 1):                         #a loop adds hidden layers (with 2 hidden_num we just have 1 hidden layer here! the article cliams 2 hidden layers)
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, z0_dim))       #final layer (maps the last hidden layer to the output layer)

    def forward(self, x):                                       # x: input data
        for layer in self.layers[:-1]:                          # excludes the last layer because the activation is not applied to the output layer (The -1 index in Python is shorthand to access the last item of a list, so self.layers[-1] refers to the final layer that maps the last hidden layer to the output layer)
            x = F.relu(layer(x))                                # applies the ReLU activation function to the output of each layer
        z0 = self.layers[-1](x)                                 # output layer
        return z0

    def reset_parameters(self):                                # reinitializing the model's weights for retraining
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class NNOPT(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim, A, B, b):
        super(NNOPT, self).__init__()
        self.A = A                                           # orthogonal projection (satisfying hard linear constriants)
        self.B = B
        self.b = b
        self.chunk = torch.mm(B.t(),                         # B.t= transpose of B
                             torch.inverse(                  # inverse of a squared matrix
                                 torch.mm(B, B.t())
                             )
                             )                               # 𝐵𝑇 (𝐵𝐵𝑇 )-1
        self.Astar = - torch.mm(self.chunk, self.A)          # -𝐵𝑇 (𝐵𝐵𝑇 )-1 * A
        self.Bstar = torch.eye(z0_dim).to(device) - torch.mm(self.chunk, self.B)   # I - 𝐵𝑇 (𝐵𝐵𝑇 )-1 * B
        self.bstar = torch.matmul(self.chunk, self.b).squeeze(-1)                  # 𝐵𝑇 (𝐵𝐵𝑇 )-1 * b    # I don't understand .squeeze(-1)   

        self.layers = nn.ModuleList()                        # a list of learnable layers for neural network
        self.layers.append(nn.Linear(input_dim, hidden_dim)) # a linear layer
        for _ in range(hidden_num - 1):                      # (_) indicate that the variable's value is not important and will not be used in the loop body.
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, z0_dim))

        self.fc_fixed1 = nn.Linear(z0_dim, z0_dim, bias=False)  ########    #fixed layers fc_fixed1 (a linear transformation with no bias and fixed weights) and fc_fixed2. 
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)  # fixed weight (not learnable- will not be updated during training), Bstar
        self.fc_fixed2 = nn.Linear(input_dim, z0_dim, bias=False)  ########    # input dimensions
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)  # Astar as weights
        self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False)    # bstar as bias

    def forward(self, x):
        x0 = x                                             # input               
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        z0 = self.layers[-1](x)                            # output after NN
        z = self.fc_fixed1(z0) + self.fc_fixed2(x0)        # output after orthogonal projection
        return z

    def reset_parameters(self):                            # reinitialize the weights of certain layers
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc_fixed1 and module is not self.fc_fixed2:
                module.reset_parameters()


class ECNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim, A, B_indep, B_dep, b):
        super(ECNN, self).__init__()
        self.A = A
        self.B_indep = B_indep
        self.B_dep = B_dep
        self.b = b

        self.B_dep_inverse = torch.inverse(B_dep)                      # B_dep should be squared. (because of inverse)
        self.Astar = - torch.mm(self.B_dep_inverse, self.A)
        self.Bstar = - torch.mm(self.B_dep_inverse, self.B_indep)
        self.bstar = torch.matmul(self.B_dep_inverse, b).squeeze(-1)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, self.B_indep.shape[1]))     # the difference is here by defining the outputs of the NN as the number of the columns of the B indep matrix.

        self.fc_fixed1 = nn.Linear(self.B_indep.shape[1], self.B_dep.shape[1], bias=False)  # the number of input is yI which is equall to the number of columns of B_indep. The output of this layer is equall to the numbr of yD, which is defined by the number of columns of B_dep.
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
        self.fc_fixed2 = nn.Linear(input_dim, self.B_dep.shape[1], bias=False)     # the number of input in this layer is the number of x (input of the NN or the input of the whole problem). The output of this layer is equall to the numbr of yD, which is defined by the number of columns of B_dep.
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)
        self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False)

    def forward(self, x):
        x0 = x                                                           # x should be defined somewhere. I don't know where it is defined. and self.
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        z_indep = self.layers[-1](x)                                    # (batch_size, self.B_indep.shape[1]), the difference here is the output of the NN is not z, it is z_indep.
        z_dep = self.fc_fixed1(z_indep) + self.fc_fixed2(x0)            # (batch_size, self.B_dep.shape[1]), despite the previous method, here we do not achieve the final output, but we derive the dependant output.
        z = torch.cat((z_dep, z_indep), dim=1)                          # (batch_size, self.B_indep.shape[1] + self.B_dep.shape[1]), join these two tensors together
        return z

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc_fixed1 and module is not self.fc_fixed2:
                module.reset_parameters()