import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from multiprocessing import cpu_count #check if we can use this
from typing import NamedTuple #check if we can use this

from dataset import GTZAN
from evaluation import evaluate

import torch.optim as optim
from torch.optim.optimizer import Optimizer

# For tensorboard
from torch.utils.tensorboard import SummaryWriter


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
    transform_grey = transforms.Grayscale(num_output_channels=3)

    GTZAN_train = GTZAN("train.pkl")
    train_loader = DataLoader(GTZAN_train.dataset, batch_size = 64, shuffle = True, num_workers = cpu_count(), pin_memory = True)

    GTZAN_test = GTZAN("val.pkl")
    test_loader = DataLoader(GTZAN_test.dataset, batch_size = 64, num_workers = cpu_count(), pin_memory = True)

    model_res = torchvision.models.resnet34(pretrained=True)
    for param in model_res.parameters():
        param.requires_grad = False

    num_features = model_res.fc.in_features
    model_res.fc = nn.Linear(num_features, 10)

    model_res = model_res.to(DEVICE)

    # model = shallow_CNN(height = 80, width = 80, channels = 1, class_count = 10)

    optimiser = optim.Adam(model_res.fc.parameters(), lr=5e-5, betas=(0.9,0.999), eps=1e-08)

    criterion = nn.CrossEntropyLoss()

    summary_writer = SummaryWriter('logs', flush_secs=5)

    trainer = Trainer(
        model_res, train_loader, criterion, DEVICE, test_loader, optimiser, summary_writer, transform_grey
    )

    test_preds, test_labels = trainer.train(epochs = 100, val_frequency = 1) # runs validated epoch+1%val_freq
    confmat = ConfusionMatrix(num_classes = 10, normalize = 'true')
    conf_matrix = confmat(test_preds, test_labels)
    print(conf_matrix)

    summary_writer.close()

# class shallow_CNN(nn.Module):
#     def __init__(self, height:int, width: int, channels: int, class_count: int):
#         super().__init__()
#         self.input_shape = SpectrogramShape(height=height, width=width, channels=channels)
#         self.class_count = class_count

#         self.conv1_LHS = nn.Conv2d(
#             in_channels = self.input_shape.channels,
#             out_channels = 16,
#             kernel_size = (10, 23),
#             padding = "same",
#         )
#         self.initialise_layer(self.conv1_LHS)

#         self.pool1_LHS = nn.MaxPool2d(
#             kernel_size = (1, 20),
#         )

#         self.conv1_RHS = nn.Conv2d(
#             in_channels = self.input_shape.channels,
#             out_channels = 16,
#             kernel_size = (21, 20),
#             padding = "same",
#         )
#         self.initialise_layer(self.conv1_RHS)

#         self.pool1_RHS = nn.MaxPool2d(kernel_size = (20, 1))

#         self.fc1 = nn.Linear(10240, 200)
#         self.initialise_layer(self.fc1)

#         self.dropout1 = nn.Dropout(0.1)

#         self.fc2 = nn.Linear(200, 10)
#         self.initialise_layer(self.fc2)

#     def forward(self, x):
#         x_LHS = self.conv1_LHS(x)
#         x_LHS = F.leaky_relu(x_LHS, 0.3)
#         x_LHS = self.pool1_LHS(x_LHS)
#         x_LHS = torch.flatten(x_LHS, 1)

#         x_RHS = self.conv1_RHS(x)
#         x_RHS = F.leaky_relu(x_RHS, 0.3)
#         x_RHS = self.pool1_RHS(x_RHS)
#         x_RHS = torch.flatten(x_RHS, 1)

#         x_merged = self.merge(x_LHS, x_RHS)
#         x = self.fc_layers(x_merged)

#         # x = F.softmax(x, 1)
#         return x

#     def merge(self, x_LHS, x_RHS):
#         x_merged = torch.cat((x_LHS, x_RHS), 1)
#         return x_merged

#     def fc_layers(self, x_merged):
#         x = self.fc1(x_merged)
#         x = F.leaky_relu(x, 0.3)
#         x = self.dropout1(x)

#         x = self.fc2(x)
#         return x

#     @staticmethod
#     def initialise_layer(layer):
#         if hasattr(layer, "bias"):
#             nn.init.zeros_(layer.bias)
#         if hasattr(layer, "weight"):
#             nn.init.kaiming_uniform_(layer.weight)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        test_loader: DataLoader,
        optimiser: Optimizer,
        summary_writer: SummaryWriter,
        transform_grey: torchvision.transforms,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimiser = optimiser
        self.summary_writer = summary_writer
        self.transform_grey = transform_grey

    def train(self, epochs: int, val_frequency: int):
        self.model.train()
        for epoch in range(0, epochs):
            print("epoch no", epoch)
            self.model.train() # Sets module in train mode
            for _, batch, labels, _ in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                # rgb_batch = self.transform_grey(batch[0])
                # print("hello")
                # print(rgb_batch.shape)
                # print("world")

                batch_repeated = batch.expand(len(batch), 3, 80, 80)
                # print(batch_repeated.size())

                output = self.model.forward(batch_repeated)

                # L1 weight regularisation
                penalty = 1e-4
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())

                loss = self.criterion(output, labels)
                loss += (penalty * l1_norm)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                # loss.backward()
                # print(temp)
                # self.optimiser.step()
                # self.optimiser.zero_grad() # For the backwards pass

                # For tensorboard output
                preds = output.argmax(-1)
                train_accuracy = self.accuracy(preds, labels) * 100
                self.summary_writer.add_scalar('accuracy/train', train_accuracy, epoch)
                self.summary_writer.add_scalar('loss/train', loss.item(), epoch)


            if ((epoch + 1) % val_frequency) == 0:
                print("In test if")
                test_preds, test_labels = self.test()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()
        
        return test_preds, test_labels

    def accuracy(self, preds, labels):
        assert len(labels) == len(preds)
        acc = float((preds == labels).sum()) / len(labels)
        return acc

    def test(self):
        preds = []
        all_labels = []
        total_loss = 0
        self.model.eval() # Sets module in evaluation

        # No need to track gradients for validation since we're not optimising
        with torch.no_grad():
            for _,batch,labels,_ in self.test_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                # rgb_batch = self.transform_grey(batch)


                batch_repeated = batch.expand(len(batch), 3, 80, 80)

                outputs = self.model(batch_repeated)

                preds.extend(list(outputs))
                all_labels.extend(list(labels))

        accuracy = evaluate(preds, "val.pkl")

        preds_tensor = torch.stack(preds)
        all_labels_tensor = torch.stack(all_labels)
        preds = preds_tensor.argmax(-1)
        return preds, all_labels_tensor

if __name__ == "__main__":
    main()
