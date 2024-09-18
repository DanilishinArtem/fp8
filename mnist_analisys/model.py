from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.lin2 = nn.Linear(128, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool(x)
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu3(self.lin1(x))
        x = self.lin2(x)
        return self.log_softmax(x)