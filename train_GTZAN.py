import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from multiprocessing import cpu_count #check if we can use this
from typing import NamedTuple #check if we can use this

from dataset import GTZAN
from evaluation import evaluate

class SpectrogramShape(NamedTuple):
    height: int
    width: int
    channels: int

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main():
    transform = transforms.ToTensor()

    GTZAN_train = GTZAN("train.pkl")
    train_loader = DataLoader(GTZAN_train.dataset, batch_size = 128, shuffle = True, num_workers = cpu_count(), pin_memory = True)
    # print(GTZAN_train.dataset.size())
    # print(train_loader.shape)
    # print(train_loader)

    model = shallow_CNN(height = 80, width = 80, channels = 1, class_count = 10)

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model, train_loader, criterion, DEVICE
    )

    trainer.train(epochs = 2)

class shallow_CNN(nn.Module):
    def __init__(self, height:int, width: int, channels: int, class_count: int):
        super().__init__()
        self.input_shape = SpectrogramShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        print("starting conv1 LHS")
        self.conv1_LHS = nn.Conv2d(
            in_channels = self.input_shape.channels, 
            out_channels = 16, 
            kernel_size = (10, 23), 
            padding = "same",
        )
        self.initialise_layer(self.conv1_LHS)

        print("starting pool1 LHS")
        self.pool1_LHS = nn.MaxPool2d(
            kernel_size = (1, 20),
        )

        print("starting conv1 RHS")
        self.conv1_RHS = nn.Conv2d(
            in_channels = self.input_shape.channels, 
            out_channels = 16, 
            kernel_size = (21, 20), 
            padding = "same",
        )
        self.initialise_layer(self.conv1_RHS)

        print("starting pool1 RHS")
        self.pool1_RHS = nn.MaxPool2d(kernel_size = (20, 1))

        print("starting fc1")
        self.fc1 = nn.Linear(10240, 200)
        self.initialise_layer(self.fc1)

        print("starting dropout")
        self.dropout1 = nn.Dropout2d(0.1)

        print("starting fc2")
        self.fc2 = nn.Linear(200, 10)
        self.initialise_layer(self.fc2)

    def forward(self, x):
        # print("starting forward")
        x_LHS = self.conv1_LHS(x)
        x_LHS = F.leaky_relu(x_LHS, 0.3)
        x_LHS = self.pool1_LHS(x_LHS)
        x_LHS = torch.flatten(x_LHS, 1)

        x_RHS = self.conv1_RHS(x)
        x_RHS = F.leaky_relu(x_RHS, 0.3)
        x_RHS = self.pool1_RHS(x_RHS)
        x_RHS = torch.flatten(x_RHS, 1)

        x_merged = self.merge(x_LHS, x_RHS)
        x = self.fc_layers(x_merged)

        x = F.softmax(x, 1)
        return x

    def merge(self, x_LHS, x_RHS):
        x_merged = torch.cat((x_LHS, x_RHS), 1)
        return x_merged

    def fc_layers(self, x_merged):
        x = self.fc1(x_merged)
        x = F.leaky_relu(x, 0.3)
        x = self.dropout1(x)

        x = self.fc2(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device, 
        # val_loader: DataLoader,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        # self.val_loader = val_loader
        self.criterion = criterion

    def train(self, epochs: int):
        self.model.train()
        for epoch in range(0, epochs):
            print("epoch no", epoch)
            self.model.train()
            for _, batch, labels, _ in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                output = self.model.forward(batch)
                print("output shape", output.shape)
                # print("output[0]", output[0])

                print("labels shape", labels.shape)
                loss = self.criterion(output, labels)
                print("loss shape", loss.shape)
                loss.backward()

                # with torch.no_grad():
                #     preds = output.argmax(-1)
                accuracy = evaluate(output, "train.pkl")
                print("accuracy", accuracy)

if __name__ == "__main__":
    main()