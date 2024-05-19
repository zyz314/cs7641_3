import torch.utils
from datasets.uciml import AdultDataset, DryBeanDataset
from models.mlp import SimpleMLP
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.autograd import Variable

import matplotlib.pyplot as plt


def train_model(model : nn.Module, dataset : Dataset):
    """
    """
    n_epochs = 30

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.01)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5)

    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    test_set = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_set = DataLoader(val_dataset, batch_size=64, shuffle=True)

    train_losses = []
    train_accuracies = []
    eval_losses = []
    eval_accuracies = []

    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            temp_losses = []
            temp_accs = []

            # Train
            model.train()
            for inputs, target in test_set:
                input_var = Variable(inputs, requires_grad=True)

                optimizer.zero_grad()

                output = model(input_var)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

                temp_losses.append(loss.item())
                temp_accs.append((torch.sum(output.argmax(dim=1) == target) / inputs.shape[0]).item())
            train_losses.append(np.mean(temp_losses))
            train_accuracies.append(np.mean(temp_accs))

            # Validation
            model.eval()
            with torch.no_grad():
                temp_losses = []
                temp_accs = []
                for inputs, target in val_set:
                    output = model(inputs)
                    loss = criterion(output, target)
                    temp_losses.append(loss.item())
                    temp_accs.append((torch.sum(output.argmax(dim=1) == target) / inputs.shape[0]).item())
                eval_losses.append(np.mean(temp_losses))
                eval_accuracies.append(np.mean(temp_accs))

            lr_scheduler.step(eval_accuracies[-1])

            pbar.set_postfix(lr = optimizer.param_groups[0]['lr'], train_loss=train_losses[-1],
                             eval_loss=eval_losses[-1], train_acc=train_accuracies[-1], eval_acc=eval_accuracies[-1])
            pbar.update()

    plt.subplot(1, 2, 1)
    plt.plot(range(n_epochs - 1), train_losses[1:], label='train_loss')
    plt.plot(range(n_epochs - 1), eval_losses[1:], label='eval_loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(n_epochs - 1), train_accuracies[1:], label='train_acc')
    plt.plot(range(n_epochs - 1), eval_accuracies[1:], label='eval_acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig('loss_curves.png')


def main():
    model = SimpleMLP(16, 7)
    dataset = DryBeanDataset()

    train_model(model, dataset)

if __name__ == '__main__':
    main()
