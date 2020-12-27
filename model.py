import torch.nn as nn
import torch

""" Optional conv block """
def conv_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        
        # Protonet.py
        nn.MaxPool2d(2),
        
        # maybe
        #nn.Dropout(p=0.3)

    )


""" TODO1a Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=60, z_dim=64, num_layer=4):
        # super().__init__()
        
        # Protonet.py based
        super(FewShotModel, self).__init__()
        
        modules = []
        # Input
        modules.append(conv_block(x_dim, hid_dim))
        
        # Sets the number of hidden layers in the model.
        for i in range(num_layer):
            modules.append(conv_block(hid_dim, hid_dim))
        
        # Output
        modules.append(conv_block(hid_dim, z_dim))

        self.encoder = nn.Sequential(*modules)

        #Skip Connection
        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(z_dim * int(400/(2**(num_layer+2))) * int(400/(2**(num_layer+2))), 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
        )
        
        self.classifier2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(512,64),
        )


    def forward(self, x):
        #return embedding_vector
        
        # Protonet.py
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier1(x)
        x2 = self.classifier2(x)
        theX = torch.cat([x, x2], dim = 1)

        return theX
