from torch.utils.data import DataLoader, random_split
from dataLoader import load_mnist_dataset
from config import Config
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
from mnist_analisys.gradientCollector import GradientCollector
from torch.cuda.amp import autocast, GradScaler
from fp8.FP8Tensor import to_fp8

def convert_gradients_to_fp8(model):
    for param in model.parameters():
        if param.grad is not None:
            # param.grad.data = to_fp8(param.grad.data).to('cuda')
            param.grad.data = param.grad.data.to(torch.torch.float8_e4m3fn).to(torch.bfloat16).to('cuda')

def convert_weights_to_fp8(model):
    for param in model.parameters():
        if param is not None:
            # param.data = to_fp8(param.data).to('cuda')
            param.data = param.data.to(torch.torch.float8_e4m3fn).to(torch.bfloat16).to('cuda')


class LearningProcess:
    def __init__(self, optimizer: optim, criterion: nn.Module, writer: SummaryWriter = None):
        self.config = Config()
        self.writer = writer
        self.train_loader, self.val_loader, self.test_loader = self.createDataset()
        self.optimizer = optimizer
        self.criterion = criterion

    def createDataset(self):
        dataset = load_mnist_dataset()
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        train_size = int(self.config.split * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
    def train(self, model: nn.Module):
        collector = GradientCollector()
        print("start training\n\n")
        total_counter = 0
        total_loss = 0
        numberFaults = 0
        correct = 0
        numPic = 0
        # hookManager.remove_hooks()
        for epoch in range(self.config.num_epochs):
            model.train()
            model = model.to('cuda')
            for batch in self.train_loader:
                total_counter += 1
                images, labels = batch["image"], batch["label"]
                images = images.to('cuda')
                labels = labels.to('cuda')
                self.optimizer.zero_grad()
                with autocast(dtype=torch.bfloat16):
                    convert_weights_to_fp8(model)
                    output = model(images)
                    loss = self.criterion(output, labels)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                numPic += len(images)
                loss.backward()
                # convert gradients to fp8
                # convert_gradients_to_fp8(model)
                # convert_weights_to_fp8(model)
                # collector.collectGradients(model, self.writer, total_counter)
                self.optimizer.step()
                total_loss += loss.item()
                self.writer.add_scalar("Loss/train", loss.item(), total_counter)
                self.writer.add_scalar("Accuracy/train", correct / numPic, total_counter)
                print("For step " + str(total_counter) + " training loss = " + str(round(total_loss / total_counter,2)) + " training accuracy = " + str(round(correct / numPic,2)))

# for param in model.parameters():
#     print(param._backward_hooks)

    def validate(self, model: nn.Module):
        total_counter = 0
        model.eval()
        val_loss = 0
        correct = 0
        batch_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                total_counter += 1
                images, labels = batch["image"], batch["label"]
                output = model(images)
                val_loss += self.criterion(output, labels).item()
                batch_loss = self.criterion(output, labels).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                self.writer.add_scalar("Loss/validation", batch_loss / self.config.batch_size, total_counter)
        val_loss /= len(self.val_loader)
        accuracy = correct / len(self.val_loader.dataset)
        print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}")

    def test(self, model: nn.Module):
        model.eval()
        test_loss = 0
        correct = 0
        counter = 0
        total_counter = 0
        batch_loss = 0
        with torch.no_grad():
            for batch in self.test_loader:
                total_counter += 1
                images, labels = batch["image"], batch["label"]
                output = model(images)
                test_loss += self.criterion(output, labels).item()
                batch_loss = self.criterion(output, labels).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                self.writer.add_scalar("Loss/test", batch_loss / self.config.batch_size, total_counter)

        test_loss /= len(self.test_loader)
        accuracy = correct / len(self.test_loader.dataset)
        print(f"Test Loss: {test_loss}, Test Accuracy: {accuracy}")
