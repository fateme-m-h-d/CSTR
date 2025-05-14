import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

device = "cuda" if torch.cuda.is_available() else "cpu"


class NN(nn.Module): #Define a vanilla NN
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim): #initialize objects of a class
        super(NN, self).__init__()
        self.layers = nn.ModuleList() #initialize the layers
        self.layers.append(nn.Linear(input_dim, hidden_dim)) #first layer
        for _ in range(hidden_num - 1): #adding the hidden layers in the neural network
            self.layers.append(nn.Linear(hidden_dim, hidden_dim)) #middle layers
        self.layers.append(nn.Linear(hidden_dim, z0_dim)) #last layer

    def forward(self, x):
        for layer in self.layers[:-1]: #looping the every layer until the last one
            x = F.relu(layer(x)) #Activation function, Rectified Linear Unit (relu)
        z0 = self.layers[-1](x) #Output, last layer multiplies the activation function
        return z0

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear): #if the module is a linear neural network, y = Ax + b
                module.reset_parameters() #reset the weights/parameters in neural network


class NNOPT(nn.Module): #Define the KKT-hPINN
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim, A, B, b):
        super(NNOPT, self).__init__()
        self.A = A #initialize A matrix
        self.B = B #initialize B matrix
        self.b = b #initialize b vector
        self.chunk = torch.mm(B.t(),
                             torch.inverse(
                                 torch.mm(B, B.t())
                             )
                             ) #B^T * (B*B^T)^-1
        self.Astar = - torch.mm(self.chunk, self.A) #A* = -B^T * (B*B^T)^-1 * A
        self.Bstar = torch.eye(z0_dim).to(device) - torch.mm(self.chunk, self.B) #B* = I - [B^T * (B*B^T)^-1 * B]
        self.bstar = torch.matmul(self.chunk, self.b).squeeze(-1) #B^T * (B*B^T)^-1 * b

        self.layers = nn.ModuleList() #initialize the layers
        self.layers.append(nn.Linear(input_dim, hidden_dim)) #first layer
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim)) #middle layers
        self.layers.append(nn.Linear(hidden_dim, z0_dim)) #last layer

        self.fc_fixed1 = nn.Linear(z0_dim, z0_dim, bias=False) #y_hat
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False) #assign B* as a weight parameter
        self.fc_fixed2 = nn.Linear(input_dim, z0_dim, bias=False)  #x_hat
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False) #assign A* as a weight parameter
        self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False) #assign b* as a bias term

    def forward(self, x):
        x0 = x #initialize x
        for layer in self.layers[:-1]:
            x = F.relu(layer(x)) #activation function
        z0 = self.layers[-1](x) #output
        z = self.fc_fixed1(z0) + self.fc_fixed2(x0) #B* * y_hat + A* * x_heat
        return z

    def reset_parameters(self): #reset the parameters if the model is not defined as KKT-hPINN
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc_fixed1 and module is not self.fc_fixed2:
                module.reset_parameters()


class ECNN(nn.Module): #Define Equality Completion NN, feed both x and half of y called y_independent
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim, A, B_indep, B_dep, b):
        super(ECNN, self).__init__()
        self.A = A
        self.B_indep = B_indep
        self.B_dep = B_dep
        self.b = b

        self.B_dep_inverse = torch.inverse(B_dep) #B_D^-1
        self.Astar = - torch.mm(self.B_dep_inverse, self.A) #B_D^-1 * A
        self.Bstar = - torch.mm(self.B_dep_inverse, self.B_indep) #B_D^-1 * B_I
        self.bstar = torch.matmul(self.B_dep_inverse, b).squeeze(-1) #B_D^-1 * b

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim)) #first layer
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim)) #middle layers
        self.layers.append(nn.Linear(hidden_dim, self.B_indep.shape[1])) #last year

        self.fc_fixed1 = nn.Linear(self.B_indep.shape[1], self.B_dep.shape[1], bias=False)  
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False) #the weight in front of y_I
        self.fc_fixed2 = nn.Linear(input_dim, self.B_dep.shape[1], bias=False)  
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False) #the weight in front of x
        self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False) #bias term

    def forward(self, x):
        x0 = x
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        z_indep = self.layers[-1](x) #y_independent
        z_dep = self.fc_fixed1(z_indep) + self.fc_fixed2(x0) #y_dependent
        z = torch.cat((z_dep, z_indep), dim=1) #concatenate y_indepent and y_dependent to get y
        return z

    def reset_parameters(self): #reset the parameters if the module is not used
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc_fixed1 and module is not self.fc_fixed2:
                module.reset_parameters()