import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import NamedTuple #check if we can use this

from dataset import GTZAN

class SpectrogramShape(NamedTuple):
    height: int
    width: int
    channels: int

def main():
    GTZAN_train = GTZAN("train.pkl")

    train_loader = torch.utils.data.DataLoader(GTZAN_train.dataset)

    # print(type(train_loader))
    # print(train_loader.shape)

    model = shallow_CNN(height = 80, width = 80, channels = 16, class_count = 10)

class shallow_CNN(nn.Module):
    def __init__(self, height:int, width: int, channels: int, class_count: int):
        super().__init__()
        self.input_shape = SpectrogramShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.conv1 = nn.Conv1d(
            in_channels = self.input_shape.channels, 
            out_channels = self.input_shape.channels, 
            kernel_size = (10, 23), 
            padding = "same",
        )

        self.initialise_layer(self.conv1)

        print(self.conv1.shape)

    def forward(self, )
    
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

if __name__ == "__main__":
    main()