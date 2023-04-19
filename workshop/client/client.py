from collections import OrderedDict

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import numpy as np
from numpy import random
import flwr as fl
import sys

import time
import logging



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(left_coordinate):
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cifar = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    x = left_coordinate  ## int number from 0 to 55500
    dataset_indexes = np.arange(x,x+150)
    random.seed(10)
    random.shuffle(dataset_indexes)
    train = list(dataset_indexes)[0:int(len(dataset_indexes)*0.8)]
    test = list(dataset_indexes)[int(len(dataset_indexes)*0.8):]

    trainset = torch.utils.data.Subset(cifar, train)
    testset = torch.utils.data.Subset(cifar, test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                shuffle=True)

    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples

def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model and data
net = Net().to(DEVICE)
# trainloader, testloader, num_examples = load_data()

class CifarClient(fl.client.NumPyClient):
    def __init__(self, left_coordinate):
        self.left_coordinate = left_coordinate
        self.trainloader, self.testloader, self.num_examples = load_data(self.left_coordinate)
    

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        logging.info("Setting params from server")
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        logging.info("Fitting on data")
        self.set_parameters(parameters)
        train(net, self.trainloader, epochs=1)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
    
if __name__ == "__main__":
    while True:
        # print(sys.argv)
        try: 
            left_coordinate_index = sys.argv.index('--arg1')
            left_coordinate = sys.argv[left_coordinate_index + 1]
            # print(left_coordinate)

            left_coordinate = int(left_coordinate)
        # local: "0.0.0.0:8080"
        # docker: "federated-learning-server-1:8080"
            client = CifarClient(left_coordinate)
            fl.client.start_numpy_client(server_address="federated-learning-server-1:8080", client=client)
            break
        except Exception as e:
            logging.warning("Could not connect to server: sleeping for 5 seconds...")
            time.sleep(5)