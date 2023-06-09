import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from multiprocessing import cpu_count
from typing import NamedTuple
import random

from dataset import GTZAN
from evaluation import evaluate

# For optimiser
import torch.optim as optim
from torch.optim.optimizer import Optimizer

# For tensorboard
from torch.utils.tensorboard import SummaryWriter

# For confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class SpectrogramShape(NamedTuple):
    height: int
    width: int
    channels: int

# uses the GPU if available, otherwise runs on CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main():

    def shallow_and_ext1():
        # Get train and test data using dataset.py
        GTZAN_train = GTZAN("train.pkl")
        GTZAN_test = GTZAN("val.pkl")

        genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        split_by_genre = {"blues": [], "classical": [], "country": [], "disco": [], "hiphop": [], "jazz": [], "metal": [], "pop": [], "reggae": [], "rock": []}

        # combining the train and test datasets
        train_counter = 0
        train_counter_next = 0
        test_counter = 0
        test_counter_next = 0
        for label in genre_list:
            train_counter_next += 1125
            split_by_genre[label] = GTZAN_train.dataset[train_counter:train_counter_next]
            train_counter = train_counter_next

            test_counter_next += 375
            split_by_genre[label].extend(list(GTZAN_test.dataset[test_counter:test_counter_next]))
            test_counter = test_counter_next

            # Shuffle the combined genre
            random.seed(10) # so that we get the same shuffle for each run
            random.shuffle(split_by_genre[label])

        """
        Splitting the data into four chunks and training four models.
        Then calculating the mean accuracy of all models.
        """
        indexes = [0,1,2,3]
        splits = [[0,375],[375,750],[750,1125],[1125,1500]]
        data = {"training":[], "test":[]}
        per_model_accuracy = []
        for i in range(4):
            print("Training model",i)
            data = {"training":[], "test":[]}
            for genre in split_by_genre:
                # Adds 25% of each genre to the test data.
                data["test"].extend(split_by_genre[genre][splits[i][0]:splits[i][1]])
                indexes.remove(i)
                # Adds remaining 75% of each genre to the training data.
                for index in indexes:
                    a = splits[index][0]
                    b = splits[index][1]
                    data["training"].extend(split_by_genre[genre][a:b])
                indexes = [0,1,2,3]
            # Training data and test data for one fold
            # Length of training data: 11,250
            # Length of testing data: 3,750

            train_loader = DataLoader(data["training"], batch_size = 32, shuffle = True, num_workers = cpu_count(), pin_memory = True)
            test_loader = DataLoader(data["test"], batch_size = 32, num_workers = cpu_count(), pin_memory = True)

            model = shallow_CNN(height = 80, width = 80, channels = 1, class_count = 10)

            # define hyperparameters for Trainer
            optimiser = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9,0.999), eps=1e-08)
            criterion = nn.CrossEntropyLoss()

            summary_writer = SummaryWriter('logs', flush_secs=5)

            # instantiates Trainer with the shallow model
            trainer = Trainer(
                model, train_loader, criterion, DEVICE, test_loader, optimiser, summary_writer
            )

            # training the model and returning the results of the validation data on the last epoch
            test_preds, test_labels = trainer.train(epochs = 100, val_frequency = 100)

            # compute model accuracy
            num_correct = (test_preds == test_labels).sum()
            accuracy_model = (num_correct/len(test_labels)) * 100
            per_model_accuracy.append(accuracy_model)

            summary_writer.close()

        # Calculate the mean accuracy of the four models
        print("All model accuracy:", per_model_accuracy)
        mean_accuracy = sum(per_model_accuracy)/4
        print("Mean accuracy:", mean_accuracy)

    def base_shallow():
        # Get train and test data using dataset.py
        GTZAN_train = GTZAN("train.pkl")
        GTZAN_test = GTZAN("val.pkl")

        train_loader = DataLoader(GTZAN_train.dataset, batch_size = 32, shuffle = True, num_workers = cpu_count(), pin_memory = True)
        test_loader = DataLoader(GTZAN_test.dataset, batch_size = 32, num_workers = cpu_count(), pin_memory = True)

        model = shallow_CNN(height = 80, width = 80, channels = 1, class_count = 10)

        # define hyperparameters for Trainer
        optimiser = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9,0.999), eps=1e-08)
        criterion = nn.CrossEntropyLoss()

        summary_writer = SummaryWriter('logs', flush_secs=5)

        # instantiates Trainer with the shallow model
        trainer = Trainer(
            model, train_loader, criterion, DEVICE, test_loader, optimiser, summary_writer
        )

        # training the model and returning the results of the validation data on the last epoch
        test_preds, test_labels = trainer.train(epochs = 100, val_frequency = 100)
        # test_preds, test_labels, pop_match, hip_hop_match, reggae_match = trainer.train(epochs = 10, val_frequency = 1) # runs validated epoch+1%val_freq

        # processing results so they can be visualised in a confusion matrix
        test_preds = list(test_preds.cpu().numpy())
        test_labels = list(test_labels.cpu().numpy())

        genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        conf_matrix = confusion_matrix(test_labels, test_preds, normalize = 'true')
        df_cm = pd.DataFrame(conf_matrix, index=[genre for genre in genre_list], columns=[genre for genre in genre_list])
        plt.figure(figsize=(10, 8))
        conf_heatmap = sns.heatmap(df_cm, annot=True)
        conf_heatmap.set(xlabel='Predicted Label', ylabel='True Label', title="100 Epoch Model")
        summary_writer.add_figure("Confusion Matrix", conf_heatmap.get_figure())

        """
        Code for generating spectrograms but only needed once.
        pop_match_spect = pop_match.cpu().numpy()
        summary_writer.add_image("Pop Spectrogram", pop_match_spect, dataformats='CHW')

        # pop_match = pop_match[-1, :, :]
        hiphop_match_spect = hip_hop_match.cpu().numpy()
        summary_writer.add_image("HipHop Spectrogram", hiphop_match_spect, dataformats='CHW')

        reggae_match_spect = reggae_match.cpu().numpy()
        summary_writer.add_image("Reggae Spectrogram", reggae_match_spect, dataformats='CHW')
        """
        summary_writer.close()

    """
    run base_shallow() for the implementation of the shallow CNN architecture.
    run shallow_and_ext1() for the shallow CNN architecture with stratified four-fold cross validation.
    """
    base_shallow()
    # shallow_and_ext1()


class shallow_CNN(nn.Module):
    def __init__(self, height:int, width: int, channels: int, class_count: int):
        """ This function initialises the layers of the CNN """
        super().__init__()
        self.input_shape = SpectrogramShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        self.conv1_LHS = nn.Conv2d(
            in_channels = self.input_shape.channels,
            out_channels = 16,
            kernel_size = (10, 23),
            padding = "same",
        )
        self.initialise_layer(self.conv1_LHS)

        self.pool1_LHS = nn.MaxPool2d(
            kernel_size = (1, 20),
        )

        self.conv1_RHS = nn.Conv2d(
            in_channels = self.input_shape.channels,
            out_channels = 16,
            kernel_size = (21, 20),
            padding = "same",
        )
        self.initialise_layer(self.conv1_RHS)

        self.pool1_RHS = nn.MaxPool2d(kernel_size = (20, 1))

        self.fc1 = nn.Linear(10240, 200)
        self.initialise_layer(self.fc1)

        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(200, 10)
        self.initialise_layer(self.fc2)

    def forward(self, x):
        """ This function passes the data through the model """
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


    """ This function is used to intialise our layers before the forward pass.
        We set the biases to be 0 and the weights are initialised from a uniform distribution.
    """
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_uniform_(layer.weight)

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
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimiser = optimiser
        self.summary_writer = summary_writer

    """ This function trains the model and evaluates the performance every {val_frequency} epochs."""
    def train(self, epochs: int, val_frequency: int):
        self.model.train()
        for epoch in range(0, epochs):
            print("epoch no", epoch)
            self.model.train() # Sets module in train mode
            total_training_loss = 0
            for _, batch, labels, _ in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                output = self.model.forward(batch)

                # L1 weight regularisation
                penalty = 1e-4
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())

                loss = self.criterion(output, labels)
                loss += (penalty * l1_norm)
                total_training_loss += loss.item()

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

            average_training_loss = total_training_loss / len(self.train_loader)
            if ((epoch + 1) % val_frequency) == 0:
                preds = output.argmax(-1)
                train_accuracy = self.accuracy(preds, labels) * 100
                print("Train accuracy:", train_accuracy)

                test_accuracy, average_test_loss, test_preds, test_labels = self.test()
                # test_accuracy, test_loss, test_preds, test_labels, pop_match, hip_hop_match, reggae_match = self.test()

                self.summary_writer.add_scalars('accuracy', {"train":train_accuracy, "test":test_accuracy}, epoch)
                self.summary_writer.add_scalars('loss', {"train":average_training_loss, "test":average_test_loss}, epoch)

                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

        return test_preds, test_labels #, pop_match, hip_hop_match, reggae_match

    # This function calculates the accuracy of the model
    def accuracy(self, preds, labels):
        assert len(labels) == len(preds)
        acc = float((preds == labels).sum()) / len(labels)
        return acc

    def test(self):
        preds = []
        all_labels = []
        total_loss = 0
        self.model.eval() # Sets module in evaluation mode
        total_loss = 0

        # No need to track gradients for validation since we're not optimising
        with torch.no_grad():
            for _,batch,labels,_ in self.test_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(batch)
                preds.extend(list(outputs))
                all_labels.extend(list(labels))

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                """
                Code for producing the spectrograms, only used once
                for i in range(0, len(batch)): #for each value in the batch
                    output = torch.argmax(outputs[i])
                    if (labels[i] == 7) and (output == 7):
                        pop_match = batch[i]
                    if (labels[i] == 4) and (output == 9):
                        hip_hop_match = batch[i]
                    if (labels[i] == 9) and (output == 4):
                        reggae_match = batch[i]
                """

        accuracy = evaluate(preds, "val.pkl")
        average_loss = total_loss / len(self.test_loader)

        preds_tensor = torch.stack(preds)
        all_labels_tensor = torch.stack(all_labels)
        preds = preds_tensor.argmax(-1)

        return accuracy, average_loss, preds, all_labels_tensor #, pop_match, hip_hop_match, reggae_match

if __name__ == "__main__":
    main()
