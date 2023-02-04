import torch
import torch.nn as nn

from torchvision.models import resnet18#, ResNet18_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def bias_init(module):
    ''' 
    We initialize bias with zeros like it's done in keras
    Params:
        m: nn.module
    '''
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        torch.nn.init.zeros_(module.bias)


class MLP_Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        '''
        Params:
            input_dim: Integer, the dimensionality of the input to the MLP
            hidden_dim: Integer, the dimensionality of the hidden layer of the MLP
        '''
        super().__init__()

        self.hidden_dim = hidden_dim
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),#.double(),
            nn.ReLU(),#.double(),
            nn.Linear(hidden_dim, hidden_dim),#.double(),
            nn.ReLU(),#.double(),
            nn.Linear(hidden_dim, hidden_dim),#.double(),
            nn.ReLU()#.double()
        )
        self.model.apply(bias_init)

    def forward(self, x):
        '''
        Params:
            x: Input Tensor to the MLP Encoder

        Return:
            out: Tensor of shape [B,hidden_dim] representing the output of the MLP Encoder
        '''
        out = self.model(x)
        
        return out

class ResNet_Encoder(nn.Module):
    def __init__(self,mlp_hidden_dim):
        '''
        Params:
            mlp_hidden_dim - Integer, the hidden dimensionality of MLP
        '''
        super().__init__()
        self.hidden_dim = mlp_hidden_dim

        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsampling=nn.Sequential(
            upsample,
            upsample,
            upsample,

        )

        model = resnet18(pretrained=True)
        #model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)#for newer versions of pytorch

        self.resnet = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.mlp = nn.Sequential(
            nn.Linear(model.fc.in_features,mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(mlp_hidden_dim,mlp_hidden_dim),
            nn.ReLU()
        )
        self.mlp.apply(bias_init)
   
    def forward(self,x):
        '''
        Params:
            x: Input tensor to the ResNetEncoder

        Return:
            out: Tensor, output of the Resnet Encoder
        '''
        #upsample the input
        out=self.upsampling(x)
        #pass the upsampled input through resnet18
        out=self.resnet(out)#batch_size,512,1,1
        #convert it to 2d
        out=out.squeeze()
        #pass through a 2-layer MLP
        out = self.mlp(out) 
        return out


class Encoder(nn.Module):
    '''
    Wrap around an encoder architecture and apply the reparameterization trick
    '''
    def __init__(self, encoder_model, latent_dim):
        '''
        Params:
            encoder_model: Encoder architecture that we wrap around.
            latent_dim: Integer, Latent Dimensionality of AutoEncoder (for mean and log_var)
        '''
        super().__init__()

        self.encoder_model = encoder_model
        hidden_dim = self.encoder_model.hidden_dim

        self.enc_param_mean = nn.Linear(hidden_dim, latent_dim)
        self.enc_param_mean.apply(bias_init)  

        self.enc_param_log_var = nn.Linear(hidden_dim, latent_dim)
        self.enc_param_log_var.apply(bias_init)  

    def reparameterization_trick(self, mean, log_var):
        '''
        Apply the reparameterization trick

        Inputs:
            mean - tensor of shape [B, latent_dim] representing the predicted mean of the latent distribution
            log_var - tensor of shape [B, latent_dim] representing the predicted log variance of the latent distribution
        '''

        epsilon = torch.randn_like(mean)
        z = torch.exp(0.5*log_var)*epsilon + mean

        return z

    def forward(self, x):
        '''
        Params:
            x: Input tensor to the encoder

        Return:
            mean: Tensor of shape [B, latent_dim] representing the predicted mean of the latent distribution
            log_var: Tensor of shape [B, latent_dim] representing the predicted log variance of the latent distribution
            z: Tensor of shape [B, latent_dim] representing the latent vectors
        '''
        encoder_out = self.encoder_model(x)
        mean = self.enc_param_mean(encoder_out)  # batch_size, latent_dim
        log_var = self.enc_param_log_var(encoder_out)  # batch_size, latent_dim
        z = self.reparameterization_trick(mean, log_var)

        return mean, log_var, z
