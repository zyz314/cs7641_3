import torch.utils
from datasets.uciml import AdultDataset, DryBeanDataset
from models.mlp import TwoLayerMLP
from tqdm import tqdm
import copy
import os

from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
from torch import nn
from skorch import NeuralNetClassifier
from skorch import NeuralNet
from pyperch.neural import BackpropModule, RHCModule
from pyperch.utils.decorators import add_to
from skorch.dataset import unpack_data
import copy
import matplotlib.pyplot as plt
import random

from skorch.callbacks import EpochScoring


def plot_training_curves(train_losses, train_accuracies, eval_losses, eval_accuracies, plot_name='loss_curves.png'):
    """
    Plot the training loss and accuracy curves
    """
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='train_loss')
    plt.plot(range(len(eval_losses)), eval_losses, label='eval_loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accuracies)), train_accuracies, label='train_acc')
    plt.plot(range(len(eval_accuracies)), eval_accuracies, label='eval_acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(plot_name)


def train_backprop_drybean(data_splits=[0.7, 0.3]) -> tuple:
    '''
    '''
    dataset = DryBeanDataset()
    # use, _ = random_split(dataset, [0.1, 0.9])
    train_dataset, test_dataset = random_split(dataset, data_splits)

    model = NeuralNetClassifier(
        # module=RHCModule,
        module=BackpropModule,
        module__input_dim=dataset.get_num_features(),
        module__output_dim=dataset.get_num_classes(),
        module__hidden_units=50,
        module__hidden_layers=1,
        module__dropout_percent=0.,
        # module__step_size=2e-3,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        # optimizer__momentum=0.9,
        optimizer__weight_decay=2e-4,
        lr=5e-4,
        max_epochs=10,
        batch_size=16,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=1,
        callbacks=[EpochScoring(
            scoring='accuracy', name='train_acc', on_train=True, lower_is_better=False),],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    # RHCModule.register_rhc_training_step()
    # model.registe

    X, y = train_dataset[:]
    X, y = X.numpy(), y.numpy()
    model.fit(X, y)

    X, y = test_dataset[:]
    X, y = X.numpy(), y.numpy()
    pred = model.predict(X)
    acc = accuracy_score(y, pred)

    print(f'Final accuracy {acc}')
    return (model, model.history[:, 'train_loss'], model.history[:, 'train_acc'],
            model.history[:, 'valid_loss'], model.history[:, 'valid_acc'])


def set_seed(seed=123456789):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # This environment variable is required for `use_deterministic_algorithms`
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def main():
    set_seed()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    best_model, train_losses, train_accuracies, eval_losses, eval_accuracies = train_backprop_drybean()
    torch.save(best_model, os.path.join(
        'checkpoints', 'drybean_bp_best_model.pt'))
    plot_training_curves(train_losses, train_accuracies, eval_losses, eval_accuracies,
                         plot_name=os.path.join('checkpoints', 'drybean_bp_loss_curves.png'))

    # best_model, train_losses, train_accuracies, eval_losses, eval_accuracies = train_mlp_adult()
    # torch.save(best_model, os.path.join(
    #     'checkpoints', 'adult_nn_best_model.pt'))
    # plot_training_curves(train_losses[1:], train_accuracies[1:], eval_losses[1:], eval_accuracies[1:],
    #                      plot_name=os.path.join('checkpoints', 'adult_nn_loss_curves.png'))


if __name__ == '__main__':
    main()
