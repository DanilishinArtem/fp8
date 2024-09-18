import torch.optim as optim
import torch.nn as nn
from model import Model
from torch.utils.tensorboard import SummaryWriter
from mnist_analisys.config import Config
from mnist_analisys.learningProcess import LearningProcess
from mnist_analisys.config import Config
import time
import torch
from mnist_analisys.faults import HookSetter

torch.manual_seed(20)

config = Config()

if __name__ == "__main__":
    model = Model().bfloat16()
    writer = SummaryWriter(config.pathToLogs + "_" + str(time.time()))
    
    HookSetter(model, writer)

    config = Config()
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    learner = LearningProcess(optimizer, criterion, writer)
    learner.train(model)
    # learner.validate(model)
    # learner.test(model)
