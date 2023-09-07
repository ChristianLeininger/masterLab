import torch
import torch.nn as nn

class EvalModel(nn.Module):
    def __init__(self, input_channels, conv_layers, fc_layers, activation_fn=nn.ReLU()):
        super(EvalModel, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        
        # Convolutional Layers
        for in_channels, out_channels, kernel_size, stride, padding in conv_layers:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            self.conv_layers.append(activation_fn)
        
        # Calculate the size of the flattened tensor after the conv layers
        self.flattened_size = self._get_flattened_size(input_channels, conv_layers)
        
        # Fully Connected Layers
        prev_layer_size = self.flattened_size
        for layer_size in fc_layers:
            self.fc_layers.append(nn.Linear(prev_layer_size, layer_size))
            self.fc_layers.append(activation_fn)
            prev_layer_size = layer_size
        
        # Output layer (single neuron for evaluation score)
        self.fc_layers.append(nn.Linear(prev_layer_size, 1))
    
    def _get_flattened_size(self, input_channels, conv_layers):
        x = torch.zeros(1, input_channels, 7, 6)  # Assuming a 7x6 Connect Four board
        for layer in self.conv_layers:
            x = layer(x)
        return x.numel()
    
    def forward(self, x):
        
        for layer in self.conv_layers:
            
            x = layer(x)
        
        x = x.view(-1)
        for layer in self.fc_layers:
            x = layer(x)
        
        return x
""" 
# Example usage
# input_channels = 3 (for one-hot encoding of empty, player1, player2)
# conv_layers = [(in_channels, out_channels, kernel_size, stride, padding)]
# fc_layers = [128, 64] (hidden layer sizes)
model1 = GenericCNN(input_channels=3, 
                   conv_layers=[(3, 16, 3, 1, 1), (16, 32, 3, 1, 1)], 
                   fc_layers=[128, 64], 
                   activation_fn=nn.ReLU())

model2 = GenericCNN(input_channels=3, 
                   conv_layers=[(3, 16, 5, 1, 2), (16, 32, 5, 1, 2)], 
                   fc_layers=[128, 64, 32], 
                   activation_fn=nn.LeakyReLU())
"""