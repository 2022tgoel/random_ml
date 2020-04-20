import numpy as np
import mnist
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.autograd import Variable

def get_data():
    images = mnist.train_images()
    labels = mnist.train_labels()
    return np.expand_dims(images, axis=1), labels #np.eye(10)[labels]

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.Linear(100, 10)
        )
        self.activate = nn.Softmax(dim=1)
    def forward(self, x, training=False):
        x = self.model(x)
        if not training:
            x = self.activate(x)
        return x

def train(num_epochs):
    for i in range(num_epochs):
        losses = []
        acc = []
        for x_batch, y_batch in train_loader:
            #send pytorch tensors to device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            #forward propogate
            model.train()
            yhat = model(x_batch, training=True)
            #cal accuracy
            pred = torch.argmax(yhat, dim=1)
            acc.append(torch.sum(pred == y_batch).item()/y_batch.numel())
            #calculate loss
            loss = loss_function(yhat, y_batch)
            #backward propogate
            loss.backward()
            #update weights and zero gradients    
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        print("Epoch %d: " % i, sum(losses)/len(losses), " Accuracy: ", sum(acc)/len(acc))

def validate():
    val_losses = []
    val_acc = []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            model.eval()
            yhat = model(x_val)
            pred = torch.argmax(yhat, dim=1)
            val_acc.append(torch.sum(pred == y_val).item()/y_val.numel())
            val_loss = loss_function(yhat, y_val)
            val_losses.append(val_loss.item())
    print("Validation Loss: ", sum(val_losses)/len(val_losses), " Accuracy: ", sum(val_acc)/len(val_acc))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x, y = get_data()
x_tensor, y_tensor = torch.from_numpy(x).float(), torch.from_numpy(y).long()
data = TensorDataset(x_tensor, y_tensor)
train_dataset, val_dataset = random_split(data, [50000, 10000])
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)
# defining the model
model = ConvolutionalNetwork().to(device)
# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-5)
# defining the loss function
loss_function = nn.CrossEntropyLoss(reduction='mean')

train(10)

validate()