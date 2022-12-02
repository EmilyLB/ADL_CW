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

from torch.optim.lr_scheduler import StepLR

class SpectrogramShape(NamedTuple):
    height: int
    width: int
    channels: int

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main():
    print("Total epochs 40, batch size 64, learning rate 1e-3, resnet18, Adam optimiser")
    all_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            torch.tensor((0.4850 + 0.4560 + 0.4060) / 3),
            torch.tensor((0.2290 + 0.2240 + 0.2250) / 3),
        )
    ])

    GTZAN_train = GTZAN("train.pkl")
    train_loader = DataLoader(GTZAN_train.dataset, batch_size = 64, shuffle = True, num_workers = cpu_count(), pin_memory = True)

    GTZAN_test = GTZAN("val.pkl")
    test_loader = DataLoader(GTZAN_test.dataset, batch_size = 64, num_workers = cpu_count(), pin_memory = True)

    model_res = torchvision.models.resnet18(pretrained=True)

    num_features = model_res.fc.in_features
    model_res.fc = nn.Linear(num_features, 10)
    model_res = model_res.to(DEVICE)

    optimiser = optim.Adam(model_res.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-08)
    exp_lr_scheduler = StepLR(optimiser, step_size=7, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    summary_writer = SummaryWriter('logs', flush_secs=5)

    trainer = Trainer(
        model_res, train_loader, criterion, DEVICE, test_loader, optimiser, summary_writer, all_transforms, exp_lr_scheduler,
    )

    test_preds, test_labels = trainer.train(epochs = 30, val_frequency = 5) # runs validated epoch+1%val_freq

    summary_writer.close()

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
        all_transforms: torchvision.transforms,
        scheduler: StepLR,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimiser = optimiser
        self.summary_writer = summary_writer
        self.all_transforms = all_transforms
        self.scheduler = scheduler

    def train(self, epochs: int, val_frequency: int):
        for epoch in range(0, epochs):
            print("epoch no", epoch)
            self.model.train() # Sets module in train mode
            for _, batch, labels, _ in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                self.optimiser.zero_grad()

                new_batch = []
                for i in range(0, len(batch)):
                    tmp = batch[i]
                    tmp = self.all_transforms(tmp)
                    new_batch.append(tmp)
                new_batch = torch.stack(new_batch)

                # batch_repeated = batch.expand(len(batch), 3, 80, 80)
                batch_repeated = new_batch.expand(len(new_batch), 3, 224, 224)
                batch_repeated = batch_repeated.to(self.device)

                # output = self.model.forward(batch_repeated)
                output = self.model(batch_repeated)

                # penalty = 1e-4
                # l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss = self.criterion(output, labels)
                # loss += (penalty * l1_norm)

                loss.backward()
                self.optimiser.step()

                # For tensorboard output
                preds = output.argmax(-1)
                train_accuracy = self.accuracy(preds, labels) * 100
                self.summary_writer.add_scalar('accuracy/train', train_accuracy, epoch)
                self.summary_writer.add_scalar('loss/train', loss.item(), epoch)

            self.scheduler.step()
            if ((epoch + 1) % val_frequency) == 0:
                print("Training accuracy", train_accuracy)
                print("Training loss", loss.item())
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

                new_batch = []
                for i in range(0, len(batch)):
                    tmp = batch[i]
                    tmp = self.all_transforms(tmp)
                    new_batch.append(tmp)
                new_batch = torch.stack(new_batch)

                batch_repeated = new_batch.expand(len(new_batch), 3, 224, 224)
                batch_repeated = batch_repeated.to(self.device)

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
