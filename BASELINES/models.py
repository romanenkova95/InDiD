import torch.nn as nn
from BASELINES.tscp import TCN

class MNISTTSCPEncoder(nn.Module):
    def __init__(self, seq_len=16, c_in=1, nb_filters=64, kernel_size=5, 
                 dilations=[1,2,4,8], nb_stacks=2, n_steps=50, code_size=10):       
        super(MNISTTSCPEncoder, self).__init__()        
        
        self.tcn_layer = TCN(in_channels=c_in, nb_filters=nb_filters, 
                             nb_stacks=nb_stacks, dilations=dilations, 
                             use_skip_connections=True, dropout_rate=0)
        
        self.fc1 = nn.Linear(nb_filters * seq_len, 2 * n_steps)  
        self.fc2 = nn.Linear(2 * n_steps, n_steps)    
        self.output_layer = nn.Linear(n_steps, code_size)           
        self.relu = nn.ReLU()
                
    def forward(self, x):
        out = x.flatten(2, 3).transpose(1, 2).float()
        out = self.tcn_layer(out)         
        out = out.flatten(1, 2) 
        out = self.relu(self.fc1(out)) 
        out = self.relu(self.fc2(out)) 
        out = self.output_layer(out)
        return out