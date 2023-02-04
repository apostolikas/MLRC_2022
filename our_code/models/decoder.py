#Pytorch
import torch
import torch.nn as nn
from torchvision.models import resnet18
#from our_code.Autoencoder import bias_init
from our_code.models.encoder import bias_init

class MLP_Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        '''
        Params:
            input_dim: Integer, the dimensionality of the input of the MLP
            hidden_dim: Integer, the dimensionality of the hidden layer of the MLP
            output_dim: Integer, the dimensionality of the output of the MLP
        '''
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.model.apply(bias_init)

    def forward(self, x):
        '''
        Params:
            x: Input tensor to the MLP decoder
        Return:
            out: Output tensor of the MLP decoder
        '''
        out = self.model(x)
        
        return out

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim,input_size):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size

        #! keras: input:bs,100
        #! bs,2,2,25
        #! bs,4,4,32
        #! bs,8,8,32
        #! bs,16,16,32
        #! bs,32,32,3
        #! bs,3072
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=25,out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1, output_padding=(1,1)), 
            #layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', output_padding=(1, 1),                    
            nn.ReLU(),
            nn.ConvTranspose2d(32,out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1, output_padding=(1,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32,out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1, output_padding=(1,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32,out_channels=3, kernel_size=(3,3), stride=(2,2), padding=1, output_padding=(1,1)),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # d1 layer of keras
        x = z.reshape(-1,int(self.latent_dim/4),2,2)
        x = self.net(x)
        # d6 layer of keras
        x = x.reshape(-1,self.input_size)
        return x