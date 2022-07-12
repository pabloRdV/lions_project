from torch import nn
import torch


class MLP(nn.Module): 
  def __init__(self, n_layers, input_dim, hidden_dim, output_dim=1):
    super(MLP, self).__init__()
    assert n_layers>=1, 'n_layers should at least be 1'
    self.name = 'mlp'
    self.args = [n_layers, input_dim, hidden_dim, output_dim]

    dim_in = [hidden_dim, input_dim]
    dim_out = [hidden_dim, output_dim]
    self.input_dim = input_dim    
    self.n_layers = n_layers

    self.lins = nn.ModuleList([nn.Linear(dim_in[int(i==0)], dim_out[int(i==n_layers-1)]) for i in range(n_layers)])
    self.relu = nn.ReLU()

  def forward(self, input):
    out = input.view(input.shape[0], self.input_dim)
    for lin in self.lins[:-1]:
      out = self.relu(lin(out))

    features = out.clone()
    out = self.lins[-1](out)
      
    return out, features
    

class CNN(nn.Module): # for CIFAR-S
    def __init__(self,num_classes=2):
        super(CNN,self).__init__()
        self.name = 'cnn'
        self.args = [num_classes]
        self.conv1   = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        #self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding='same')
        self.conv2   = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        self.conv3   = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same')
        self.conv4   = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        self.relu    = nn.ReLU()
        #self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3)

        self.fc1 = nn.Linear(512, 128)
        #self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(128,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        #x = self.conv2(x)
        #x = self.relu(x)  
        x = self.maxpool(x) 
        features = x.clone().view(x.size(0), -1)

        #x = self.conv3(x)
        #x = self.relu(x)
        #x = self.conv4(x)
        #x = self.relu(x)  
        #x = self.maxpool(x) 

        x = self.fc1(x.view(x.size(0), -1))
        x = self.relu(x)  
        x = self.fc2(x)

        return x, features

class CNN2(nn.Module): # with labels as input too, virer probably
    def __init__(self,num_classes=2):
        super(CNN2,self).__init__()
        self.name = 'cnn2'
        self.args = [num_classes]
        self.conv   = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.relu    = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3)

        self.fc1 = nn.Linear(512+1, 128)
        self.fc2 = nn.Linear(128,num_classes)

    def forward(self, x, y):
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x) 

        x = x.reshape(x.size(0), -1)
        y = y.reshape(y.size(0), 1)
        x = torch.cat((x, y), dim=1)
        features = x.clone()
        x = self.fc1(x)
        x = self.relu(x)  
        x = self.fc2(x)

        return x, features


class CNN3(nn.Module):  # for CelebA
    def __init__(self,num_classes=2):
        super(CNN3,self).__init__()
        self.name = 'cnn3'
        self.args = [num_classes]
        self.conv1   = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.conv2   = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.relu    = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=5)

        self.fc1 = nn.Linear(32*20, 256)
        self.fc2 = nn.Linear(256,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)  
        x = self.maxpool(x) 
        features = x.clone().view(x.size(0), -1)

        x = self.fc1(x.view(x.size(0), -1))
        x = self.relu(x)  
        x = self.fc2(x)

        return x, features
