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
from pyperch.neural import BackpropModule, RHCModule, SAModule, GAModule
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
    plt.close()


def get_bp_model(input_dim, output_dim, num_epochs):
    model = NeuralNetClassifier(
        module=BackpropModule,
        module__input_dim=input_dim,
        module__output_dim=output_dim,
        module__hidden_units=50,
        module__hidden_layers=1,
        module__dropout_percent=0.,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
        optimizer__momentum=1e-6,
        optimizer__weight_decay=0.1,
        lr=1.0,
        max_epochs=num_epochs,
        batch_size=16,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[EpochScoring(
            scoring='accuracy', name='train_acc', on_train=True, lower_is_better=False),],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    return model


def get_sa_model(input_dim, output_dim, num_epochs):
    model = NeuralNetClassifier(
        module=SAModule,
        module__input_dim=input_dim,
        module__output_dim=output_dim,
        module__hidden_units=50,
        module__hidden_layers=1,
        module__dropout_percent=0.,
        module__t=20000,
        module__cooling=.99,
        module__step_size=0.1,
        max_epochs=num_epochs,
        batch_size=16,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[EpochScoring(
            scoring='accuracy', name='train_acc', on_train=True, lower_is_better=False),],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    SAModule.register_sa_training_step()
    return model


def get_rhc_model(input_dim, output_dim, num_epochs):
    model = NeuralNetClassifier(
        module=RHCModule,
        module__input_dim=input_dim,
        module__output_dim=output_dim,
        module__hidden_units=50,
        module__hidden_layers=1,
        module__dropout_percent=0.,
        module__step_size=0.1,
        max_epochs=num_epochs,
        batch_size=16,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[EpochScoring(
            scoring='accuracy', name='train_acc', on_train=True, lower_is_better=False),],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    RHCModule.register_rhc_training_step()
    return model


def get_ga_model(input_dim, output_dim, num_epochs):
    model = NeuralNetClassifier(
        module=GAModule,
        module__input_dim=input_dim,
        module__output_dim=output_dim,
        module__hidden_units=50,
        module__hidden_layers=1,
        module__dropout_percent=0.,
        module__population_size=300,
        module__to_mate=150,
        module__to_mutate=30,
        module__step_size=0.1,
        criterion=nn.CrossEntropyLoss(),
        max_epochs=num_epochs,
        batch_size=16,
        # device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[EpochScoring(
            scoring='accuracy', name='train_acc', on_train=True, lower_is_better=False),],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    GAModule.register_ga_training_step()
    return model


def train_adult(model_generator, data_splits=[0.7, 0.3], use_pct=0.1, num_epochs=100) -> tuple:
    '''
    '''
    n_iterations = 10

    dataset = AdultDataset()

    best_model = None
    best_acc = None

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    accs = []

    for _ in range(n_iterations):
        useset, _ = random_split(dataset, [use_pct, 1.0 - use_pct])
        train_dataset, test_dataset = random_split(useset, data_splits)

        model = model_generator(dataset.get_num_features(),
                                dataset.get_num_classes(), num_epochs)

        X, y = train_dataset[:]
        X, y = X.numpy(), y.numpy()
        model.fit(X, y)

        X, y = test_dataset[:]
        X, y = X.numpy(), y.numpy()
        pred = model.predict(X)
        acc = accuracy_score(y, pred)

        train_losses.append(model.history[:, 'train_loss'])
        train_accs.append(model.history[:, 'train_acc'])
        valid_losses.append(model.history[:, 'valid_loss'])
        valid_accs.append(model.history[:, 'valid_acc'])
        accs.append(acc)

        if best_acc is None or acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)

    print(f'Final accuracy {np.average(accs)}')
    return (best_model, np.average(train_losses, axis=1), np.average(train_accs, axis=1),
            np.average(valid_losses, axis=1), np.average(valid_accs, axis=1))


def set_seed(seed=123456789):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # This environment variable is required for `use_deterministic_algorithms`
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def main():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    models = [get_bp_model, get_sa_model, get_rhc_model, get_ga_model]
    names = ['bp', 'sa', 'rhc', 'ga']

    for model, name in zip(models, names):
        set_seed()
        best_model, train_losses, train_accuracies, eval_losses, eval_accuracies = train_adult(model_generator=model,
                                                                                               use_pct=.1,
                                                                                               num_epochs=2)
        plot_training_curves(train_losses, train_accuracies, eval_losses, eval_accuracies,
                             plot_name=os.path.join('checkpoints', f"adult_{name}_loss_curves.png"))


if __name__ == '__main__':
    main()
