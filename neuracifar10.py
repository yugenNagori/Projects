import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython.display import clear_output


class ProgressPlotter:
    def __init__(self) -> None:
        self._history_dict = defaultdict(list)

    def add_scalar(self, tag: str, value) -> None:
        self._history_dict[tag].append(value)

    def display_keys(self, ax, tags):
        if isinstance(tags, str):
            tags = [tags]
        history_len = 0
        ax.grid()
        for key in tags:
            ax.plot(self._history_dict[key], marker="X", label=key)
            history_len = max(history_len, len(self.history_dict[key]))
        if len(tags) > 1:
            ax.legend(loc="lower left")
        else:
            ax.set_ylabel(key)
        ax.set_xlabel("epoch")
        ax.set_xticks(np.arange(history_len))
        ax.set_xticklabels(np.arange(history_len))

    def display(self, groups=None):
        clear_output()
        n_groups = len(groups)
        fig, ax = plt.subplots(n_groups, 1, figsize=(12, 3 * n_groups))
        if n_groups == 1:
            ax = [ax]
        for i, keys in enumerate(groups):
            self.display_keys(ax[i], keys)
        fig.tight_layout()
        plt.show()

    @property
    def history_dict(self):
        return dict(self._history_dict)
    

import random


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_random_seed(42)

import torchvision
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_val_data = CIFAR10(root="./CIFAR10", train=True, download=True, transform=transform)
test_data = CIFAR10(root="./CIFAR10", train=False, download=True, transform=transform)

train_data, val_data = random_split(train_val_data, lengths=[40000, 10000])


class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.layers= nn.Sequential(
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
            )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits
    

batch_size = 256

train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2
)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)


score_function = torchmetrics.Accuracy(task="multiclass", num_classes=10)


def train_loop(dataloader, model, criterion, optimizer, score_function, device):
    num_batches = len(dataloader)

    train_loss = 0

    for imgs, labels in dataloader:
        pred = model(imgs.to(device))
        loss =  criterion(pred, labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        train_loss += loss.detach().item()

        score_function.update(pred.cpu(), labels)

    train_loss /= num_batches

    train_score = score_function.compute().item()
    score_function.reset()

    return train_loss, train_score

def val_loop(dataloader, model, criterion, score_function, device):
    num_batches = len(dataloader)

    val_loss = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            pred =  model(imgs.to(device))
            loss =  criterion(pred, labels.to(device))

            val_loss += loss.item()

            score_function.update(pred.cpu(), labels)

    val_loss /= num_batches

    val_score = score_function.compute().item()
    score_function.reset()

    return val_loss, val_score



def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    score_function,
    device,
    num_epochs=10,
):
    pp = ProgressPlotter()
    for i in range(num_epochs):

        train_loss, train_score =  train_loop(train_loader, model, criterion, optimizer, score_function, device)


        val_loss, val_score =  val_loop(val_loader, model, criterion, score_function, device)

        pp.add_scalar("loss_train", train_loss)
        pp.add_scalar("score_train", train_score)

        pp.add_scalar("loss_val", val_loss)
        pp.add_scalar("score_val", val_score)

        pp.display([["loss_train", "loss_val"], ["score_train", "score_val"]])
    return pp
if __name__ == "__main__":
    model = FCNet().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    pp = train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    score_function,
    device,
    num_epochs=10,
    )

    accuracy = pp.history_dict["score_val"][-1]
    print(f"Accuracy {accuracy:.2f}")
print("конец")