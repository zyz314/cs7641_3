import torch.utils
from datasets.uciml import AdultDataset, DryBeanDataset
from models.mlp import TwoLayerMLP
from tqdm import tqdm
import copy
import os

from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score, train_test_split
import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skorch import NeuralNetClassifier
from skorch import NeuralNet
from skorch.dataset import ValidSplit
from pyperch.neural import BackpropModule, RHCModule, SAModule, GAModule
from pyperch.utils.decorators import add_to
from skorch.dataset import unpack_data
import copy
import matplotlib.pyplot as plt
import random

from skorch.callbacks import EpochScoring, LRScheduler


def plot_training_curves(train_losses, train_accuracies, eval_losses, eval_accuracies, plot_name='loss_curves.png'):
    """
    Plot the training loss and accuracy curves
    """
    fig, _ = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(8, 6))
    fig.supxlabel('Epoch')
    fig.tight_layout(pad=1.1, rect=(0.9, 1, 0.9, 1))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='train_loss')
    plt.plot(range(len(eval_losses)), eval_losses, label='eval_loss')
    plt.legend()
    plt.grid(visible=True)
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accuracies)), train_accuracies, label='train_acc')
    plt.plot(range(len(eval_accuracies)), eval_accuracies, label='eval_acc')
    plt.legend()
    plt.grid(visible=True)
    plt.title('Accuracy')
    plt.savefig(plot_name)
    plt.close()


def get_bp_model(input_dim, output_dim, num_epochs):
    model = NeuralNetClassifier(
        module=BackpropModule,
        module__input_dim=input_dim,
        module__output_dim=output_dim,
        module__hidden_units=10,
        module__hidden_layers=1,
        module__dropout_percent=0.,
        module__activation=nn.Tanh(),
        train_split=ValidSplit(cv=3, stratified=True),
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
        optimizer__momentum=2e-4,
        lr=0.01,
        max_epochs=num_epochs,
        batch_size=8,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[EpochScoring(
            scoring='accuracy', name='train_acc', on_train=True, lower_is_better=False)],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    p_grid = {'lr': np.logspace(-4, 1, 5),
              'optimizer__momentum': np.logspace(-1, 0, 5)}
    return model, p_grid


def get_sa_model(input_dim, output_dim, num_epochs):
    model = NeuralNetClassifier(
        module=SAModule,
        module__input_dim=input_dim,
        module__output_dim=output_dim,
        module__hidden_layers=0,
        module__dropout_percent=0.,
        module__t=100000.0,
        module__cooling=0.99,
        module__step_size=100.,
        module__activation=nn.Identity(),
        train_split=ValidSplit(cv=3, stratified=True),
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        optimizer__weight_decay=3.16227,
        max_epochs=num_epochs,
        batch_size=8,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[EpochScoring(
            scoring='accuracy', name='train_acc', on_train=True, lower_is_better=False)],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    SAModule.register_sa_training_step()

    p_grid = {'module__step_size': np.logspace(-4, 2, 3),
              'module__t': np.logspace(1, 5, 3),
              'module__cooling': np.logspace(-3, 0, 3)}
    return model, p_grid


def get_rhc_model(input_dim, output_dim, num_epochs):
    model = NeuralNetClassifier(
        module=RHCModule,
        module__input_dim=input_dim,
        module__output_dim=output_dim,
        module__hidden_layers=0,
        module__dropout_percent=0.,
        module__step_size=316227.7660168379,
        module__activation=nn.Identity(),
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam,
        optimizer__weight_decay=3.16227,
        max_epochs=num_epochs,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[EpochScoring(
            scoring='accuracy', name='train_acc', on_train=True, lower_is_better=False),],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    RHCModule.register_rhc_training_step()

    p_grid = {'module__step_size': np.logspace(4, 6, 5, base=10)}
    return model, p_grid


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
        module__activation=nn.Identity(),
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam,
        optimizer__weight_decay=3.16227,
        max_epochs=num_epochs,
        batch_size=16,
        # device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[EpochScoring(
            scoring='accuracy', name='train_acc', on_train=True, lower_is_better=False),],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    GAModule.register_ga_training_step()

    p_grid = {'module__step_size': np.logspace(-4, 2, 3),
              'module__population_size': np.logspace(2, 5, 3, dtype=int),
              'module__to_mate': np.logspace(2, 5, 3, dtype=int),
              'module__to_mutate': np.logspace(1, 2, 3, dtype=int)}
    return model, p_grid


def train(dataset, model_generator, data_splits=[0.7, 0.3], use_pct=0.1, num_epochs=100, n_iterations=10) -> tuple:
    '''
    '''
    best_model = None
    best_acc = None

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    accs = []

    useset, _ = random_split(dataset, [use_pct, 1.0 - use_pct])
    for _ in range(n_iterations):
        train_dataset, test_dataset = random_split(useset, data_splits)

        model, _ = model_generator(dataset.get_num_features(),
                                   dataset.get_num_classes(), num_epochs)
        model.set_params(verbose=1)

        try:
            X, y = train_dataset[:]
            X, y = X.numpy(), y.numpy()
        except:
            X, y = X.detach().numpy(), y.detach().numpy()
        model.fit(X, y)

        try:
            X, y = test_dataset[:]
            X, y = X.numpy(), y.numpy()
        except:
            X, y = X.detach().numpy(), y.detach().numpy()
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


def param_search(dataset, model_generator, num_epochs=50, p_grid={}, n_splits=3, use_pct=0.3, test_size=0.3, n_jobs=-1, verbose=2):
    """
    """
    use_set, _ = random_split(dataset, [use_pct, 1.0 - use_pct])
    train_set, test_set = random_split(use_set, [1.0 - test_size, test_size])

    model, p_grid = model_generator(dataset.get_num_features(),
                                    dataset.get_num_classes(), num_epochs)
    model.set_params(train_split=False, verbose=0)

    X, y = train_set[:]
    X = X.numpy()
    y = y.numpy()

    cv = StratifiedShuffleSplit(
        n_splits=3, test_size=test_size, random_state=0)
    gs = GridSearchCV(model, param_grid=p_grid, cv=cv,
                      scoring='accuracy', refit=True, n_jobs=n_jobs, verbose=verbose)
    gs.fit(X, y)

    nested_score = cross_val_score(
        gs.best_estimator_, X=X, y=y, cv=n_splits, n_jobs=n_jobs, verbose=verbose)
    print(nested_score, np.mean(nested_score))
    print(gs.best_params_)
    best_model = copy.deepcopy(gs.best_estimator_)

    X, y = test_set[:]
    X = X.numpy()
    y = y.numpy()
    pred = best_model.predict(X)

    acc = accuracy_score(y, pred)
    return best_model, acc


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

    # models = [get_bp_model, get_sa_model, get_rhc_model, get_ga_model]
    # names = ['bp', 'sa', 'rhc', 'ga']
    # n_epochs = [50, 50, 50, 50]

    search = False
    use_pct = 0.2
    models = [get_bp_model]
    names = ['bp']
    n_epochs = [150]

    dataset = DryBeanDataset(transforms=nn.BatchNorm1d(num_features=16))
    for model, name, epoch_count in zip(models, names, n_epochs):
        set_seed()
        if search:
            _ = param_search(dataset=dataset, model_generator=model,
                             use_pct=use_pct, num_epochs=epoch_count)
        else:
            _, train_losses, train_accuracies, eval_losses, eval_accuracies = train(dataset=dataset, model_generator=model,
                                                                                    use_pct=use_pct,
                                                                                    num_epochs=epoch_count)
            plot_training_curves(train_losses, train_accuracies, eval_losses, eval_accuracies,
                                 plot_name=os.path.join('checkpoints', f"drybean_{name}_loss_curves.png"))


if __name__ == '__main__':
    main()
