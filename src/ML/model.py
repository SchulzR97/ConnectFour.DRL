import torch

class Model(torch.nn.Module):
    def __init__(self, out_dim):
        torch.nn.Module.__init__(self)
        
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(4,4))
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(2,2))
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(2,2))
        self.flatten = torch.nn.Flatten(start_dim=0)
        self.flatten_batch = torch.nn.Flatten(start_dim=1)
        self.fc = torch.nn.Linear(192, 500)
        self.read_out = torch.nn.Linear(500, out_dim)
        #self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        if len(x.shape) > 3:
            x = self.flatten_batch(x)
        else:
            x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.read_out(x)
        #x = self.dropout(x)
        #x = self.softmax(x)
        x = self.dropout(x)
        return x