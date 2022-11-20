import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from multiprocessing import cpu_count #check if we can use this
from typing import NamedTuple #check if we can use this

from dataset import GTZAN

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

    trainer = Trainer(
        model, train_loader, DEVICE
    )

    trainer.train(epochs = 10)

class shallow_CNN(nn.Module):
    def __init__(self, height:int, width: int, channels: int, class_count: int):
        super().__init__()
        self.input_shape = SpectrogramShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        print("starting conv1")
        self.conv1 = nn.Conv2d(
            in_channels = self.input_shape.channels, 
            out_channels = 16, 
            kernel_size = (10, 23), 
            padding = "same",
        )
        self.initialise_layer(self.conv1)

        print("starting pool1")
        self.pool1 = nn.MaxPool2d(
            kernel_size = (1, 20),
        )

    def forward(self, x):
        # print("starting forward")
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.3)
        x = self.pool1(x)
        x = torch.flatten(x, 1)

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
        device: torch.device
        # val_loader: DataLoader,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        # self.val_loader = val_loader

    def train(self, epochs: int):
        self.model.train()
        for epoch in range(0, epochs):
            print("epoch no", epoch)
            self.model.train()
            for _, batch, labels, _ in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                output = self.model.forward(batch)
                print(output.shape)

if __name__ == "__main__":
    main()