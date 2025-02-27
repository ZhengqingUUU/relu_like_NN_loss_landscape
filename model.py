import torch
import torch.nn as nn


class two_layer_net(nn.Module):

    def __init__(self, input_size ,hidden_size, output_size, beta_1, beta_2, random_seed = None, alpha = 1, activation = "relu") -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size, bias = False)
        self.width = hidden_size
        if activation.lower() == "relu":
            self.activation = nn.ReLU(True)
        else:
            raise NotImplementedError
        self.output_layer = nn.Linear(hidden_size, output_size, bias = False)
        self.alpha = alpha
        self.weight_init(beta_1, beta_2, random_seed = random_seed)
    
    def forward(self, x_1 ):
        s_2 = self.input_layer(x_1)
        x_2 = self.activation(s_2)
        o = self.output_layer(x_2)
        return (1/self.alpha)*o
    
    def weight_init(self, beta_1, beta_2, random_seed = None):
        print()
        if random_seed != None:
            torch.manual_seed(random_seed)
        for name, layer in dict(self.named_modules()).items():
            if name == 'input_layer':
                layer.weight.data.normal_(0) ## default std to be 1
                layer.weight.data = layer.weight.data*beta_2
            elif name == 'output_layer':
                layer.weight.data.normal_(0)## default std to be 1
                layer.weight.data = layer.weight.data*beta_1