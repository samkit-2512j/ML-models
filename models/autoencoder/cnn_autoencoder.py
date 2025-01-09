import torch 
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CNNAutoEncoder(nn.Module):
    def __init__(self, in_channels, kernels, layers,activation='relu', optimizer='adam', lr=0.001,epochs = 10 ):
        super(CNNAutoEncoder,self).__init__()
        self.in_channels = in_channels
        self.kernels = kernels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.fc_layers = fc_layers
        self.activation= activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr = lr
        
        self.encoder_layers = nn.ModuleList()

        for i in range(1,len(kernels)):
            pad = (kernels[i][1] - 1)//2
            self.encoder_layers.append(nn.Conv2d(in_channels=kernels[i-1][0], out_channels=kernels[i][0],
                                                 kernel_size=kernels[i][1], stride=kernels[i][2], padding=pad))
            
        self.encoder_fc_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.encoder_fc_layers.append(nn.Linear(layers[i], layers[i + 1]))

        layers.reverse()
        # kernels.reverse()

        self.decoder_layers = nn.ModuleList()

        for i in range(len(kernels) - 1,0,-1):
            op = 0 if kernels[i][2]==1 else 1
            pad = (kernels[i][1] - 1)//2
            self.decoder_layers.append(nn.ConvTranspose2d(in_channels=kernels[i][0], out_channels=kernels[i-1][0],
                                             kernel_size=kernels[i][1],stride=kernels[i][2], padding=pad, output_padding= op))

        self.decoder_fc_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.decoder_fc_layers.append(nn.Linear(layers[i], layers[i + 1]))

        layers.reverse()            

    def activation_layer(self, x):
        if self.activation == "relu":
            return F.relu(x) 
        elif self.activation == "sigmoid":
            return F.sigmoid(x)
        elif self.activation == "tanh":
            return F.tanh(x)
        else:
            return x
        
    def get_optimizer(self):
        # if self.optimizer == "sgd":
        #     return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        # elif self.optimizer == "rmsprop":
        #     return optim.RMSprop(self.parameters(), lr=self.lr)
        # else:
        return optim.Adam(self.parameters(), lr=self.lr)
        
    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
            x = self.activation_layer(x)

        self.encoded_shape = x.shape    
        x = torch.flatten(x,start_dim=1)
        self.flattened = x.shape
        
        for i in range(len(self.encoder_fc_layers)):
            x = self.encoder_fc_layers[i](x)
            x = self.activation_layer(x)

        return x
        
    
    def decode(self, x):
        # for layer in self.decoder_fc_layers:
        #     x = layer(x)
        #     x = self.activation_layer(x)
        for i in range(len(self.decoder_fc_layers)):
            x = self.decoder_fc_layers[i](x)
            x = self.activation_layer(x)

        total = x.shape[1]
        n = x.shape[0]
        # variable = total // (self.latent_dim * self.latent_dim)
        # print(n,total,variable)
        x = x.view(n, self.encoded_shape[1], self.encoded_shape[2], self.encoded_shape[3])
        
        # print(result.shape)
        for layer in self.decoder_layers:
            x = layer(x)
            x = self.activation_layer(x)

        return x
            
    def forward(self,x):
        encoder = self.encode(x)
        decoder = self.decode(encoder)

        return decoder

        
    def train_model(self, trainLoader, valLoader):
        
        print("Using device:",self.device)
        self.to(self.device)
        optimizer = self.get_optimizer()
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0
            
            for i, (images, labels) in enumerate(trainLoader):
                optimizer.zero_grad()
                images = images.to(self.device).float()
                labels = labels.to(self.device).float()


                outputs = self(images).float()
                loss = criterion(images ,outputs)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # epoch_loss = epoch_loss / len(trainLoader)
            epoch_loss = self.eval_val(trainLoader)
            train_losses.append(epoch_loss)

            
            epoch_val_loss = self.eval_val(valLoader)
            val_losses.append(epoch_val_loss)
            
            print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {epoch_loss:.4f}, '
                f'Val Loss: {epoch_val_loss:.4f}')
        
            # if epoch_val_loss < best_val_loss - min_delta:
            #     best_val_loss = epoch_val_loss
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            #     print(f'Early stopping counter: {patience_counter}/{patience}')
                
            #     if patience_counter >= patience:
            #         if epoch+1<self.epochs:
            #             print(f'\nEarly stopping triggered after epoch {epoch+1}')
            #         break
        
        return train_losses,val_losses
    
    def eval_val(self, loader):
        self.eval()
        val_loss = 0.0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device).float()
                batch_y = batch_y.to(self.device).float()

                outputs = self(batch_X).float()
                loss = criterion(batch_X, outputs)
                val_loss += loss.item()
            # print(len(loader),val_loss)
            
        return val_loss/ len(loader)
            
        
    