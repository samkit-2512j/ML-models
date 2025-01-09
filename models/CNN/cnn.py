import torch
import numpy as np
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class cnn(nn.Module):
    def __init__(self, input_channels=1, 
                 conv_layers=[32,64], 
                 kernel_size=[5,5], 
                 dense_layers=[128,32],
                 input_size=(128,128),
                 activation = 'relu',
                 model="classifier",
                 num_classes=4
                 ):
        super(cnn, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

        self.conv_layers = nn.ModuleList()
        self.kernel_sizes = kernel_size
        self.batchnorm_layers = nn.ModuleList()
        self.activation = activation
        
        self.conv_layers.append(nn.Conv2d(in_channels=input_channels,
                                          out_channels=conv_layers[0],
                                          kernel_size=kernel_size[0],
                                          padding='same'))
        self.batchnorm_layers.append(nn.BatchNorm2d(conv_layers[0]))

        for i in range(1,len(conv_layers)):
            self.conv_layers.append(nn.Conv2d(in_channels=conv_layers[i-1],
                                              out_channels=conv_layers[i],
                                              kernel_size=kernel_size[i],
                                              padding='same'))
            self.batchnorm_layers.append(nn.BatchNorm2d(conv_layers[i]))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        h, w = input_size

        for _ in range(len(conv_layers)):
            h = h//2
            w = w//2

        flattened_size = conv_layers[-1] * h* w

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(in_features=flattened_size,
                                        out_features=dense_layers[0],
                                        ))
        
        for i in range(1,len(dense_layers)):
            self.fc_layers.append(nn.Linear(in_features=dense_layers[i-1],
                                            out_features=dense_layers[i]))
            
        if self.model == 'classifier':
            self.output_layer = nn.Linear(dense_layers[-1], num_classes)
        else:
            self.output_layer = nn.Linear(dense_layers[-1],1)

        self.dropout = nn.Dropout(0.5)

        self.to(self.device)

    def activation_layer(self, x):
        if self.activation == "relu":
            return F.relu(x) 
        elif self.activation == "sigmoid":
            return F.sigmoid(x)
        elif self.activation == "tanh":
            return F.tanh(x)
        else:
            return x

    def forward(self, x):
        feature_maps = []
        x=x.to(self.device)

        for conv,bn in zip(self.conv_layers, self.batchnorm_layers):
            x = conv(x)
            x = bn(x)
            x = self.activation_layer(x)
            x = self.pool(x)
            feature_maps.append(x)

        x = x.view(x.size(0), -1)

        for i, dense in enumerate(self.fc_layers):
            x = dense(x)
            x = self.activation_layer(x)
            if i < len(self.fc_layers) - 1:
                x = self.dropout(x)

        x = self.output_layer(x)
        self.featuremaps = feature_maps

        return x
    
    def get_loss_function(self):
        if self.model == 'classifier':
            return nn.CrossEntropyLoss()
        
        else:
            return nn.MSELoss()
        
    def train_model(self, trainLoader, valLoader, epochs=10, learning_rate=0.01, 
                patience=5, min_delta=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = self.get_loss_function()

        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            
            for i, (images, labels) in enumerate(trainLoader):
                optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.model == 'regressor':
                    labels = labels.float()

                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(trainLoader)
            train_losses.append(epoch_loss)
            
            epoch_val_loss, correct, total = self.eval_val(valLoader)
            val_losses.append(epoch_val_loss)
            
            if self.model == 'classifier':
                accuracy = 100 * correct / total
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, '
                    f'Val Loss: {epoch_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
            else:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, '
                    f'Val Loss: {epoch_val_loss:.4f}')
            
            if epoch_val_loss < best_val_loss - min_delta:
                best_val_loss = epoch_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f'Early stopping counter: {patience_counter}/{patience}')
                
                if patience_counter >= patience:
                    if epoch+1<epochs:
                        print(f'\nEarly stopping triggered after epoch {epoch+1}')
                    break
        
        return train_losses,val_losses
    
    def eval_val(self, loader):
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        criterion = self.get_loss_function()
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                if self.model != 'classifier':
                    batch_y = batch_y.view(-1, 1)
                
                if self.model == 'regressor':
                    batch_y = batch_y.float()

                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                if self.model == 'classifier':
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

        return val_loss/len(loader), correct, total
    def visualize_feature_map(self, image):
        # Set model to evaluation mode
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        image = image.unsqueeze(0).to(device)  # Add batch dimension

        # Forward pass to get feature maps
        feature_maps = []
        x = image
        with torch.no_grad():
            self.forward(x)
        feature_maps = self.featuremaps            

        # Plot feature maps
        num_layers = len(feature_maps)
        num_cols = 4  # Set number of columns for subplot layout
        fig, axes = plt.subplots(num_layers, num_cols, figsize=(12, num_layers * 3))

        for layer_idx, fmap in enumerate(feature_maps):
            fmap = fmap.cpu().detach().numpy()  # Move to CPU and convert to NumPy array
            num_features = fmap.shape[1]  # Number of feature maps

            for i in range(min(num_features, 8)):  # Plot up to 8 feature maps
                ax = axes[layer_idx, i] if num_layers > 1 else axes[i]  # Handle single-layer case
                ax.imshow(fmap[0, i], cmap='viridis')  # Show the first sample in the batch
                ax.axis('off')
                ax.set_title(f'Layer {layer_idx + 1} - Feature Map {i + 1}')

            # Turn off any remaining axes for this row
            for j in range(i + 1, num_cols):
                if num_layers > 1:
                    axes[layer_idx, j].axis('off')
                else:
                    axes[j].axis('off')

        plt.tight_layout()
        plt.show()