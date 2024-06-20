from datasets.uciml import AdultDataset, DryBeanDataset
import numpy as np
import os
import pickle
import copy

from torch import manual_seed, use_deterministic_algorithms
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import random

from tqdm import tqdm
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, HalvingGridSearchCV, cross_val_score, StratifiedShuffleSplit, ShuffleSplit
from skorch import NeuralNetClassifier
from skorch import NeuralNet
from pyperch.neural import BackpropModule, RHCModule, SAModule, GAModule

import torch
import torch.nn as nn
import torch.optim as optim
from skorch.callbacks import EpochScoring


def plot_training_curves(eval_accuracies, plot_name='loss_curves.png'):
    """
    Plot the training loss and accuracy curves
    """
    plt.plot(range(len(eval_accuracies)), eval_accuracies, label='eval_acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(plot_name)


def calculate_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate accuracy of prediction
    """
    return (np.sum(pred == target) / target.shape[0]).item()


def train_svm_drybean(kernel='linear', p_grid={}, n_splits=3, test_size=0.3, n_jobs=-1, verbose=0):
    """
    """
    dataset = DryBeanDataset()
    use_set, _ = random_split(dataset, [0.5, 0.5])
    train_set, test_set = random_split(use_set, [1.0 - test_size, test_size])

    model = NeuralNetClassifier(
        # module=RHCModule,
        module=SAModule,
        # module=GAModule,
        # module=BackpropModule,
        module__input_dim=dataset.get_num_features(),
        module__output_dim=dataset.get_num_classes(),
        module__hidden_units=50,
        module__hidden_layers=1,
        module__dropout_percent=0.,
        # module__step_size=2e-3,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        # optimizer__momentum=0.9,
        # lr=1e-2,
        # max_epochs=50,
        # batch_size=16,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[EpochScoring(
            scoring='accuracy', name='train_acc', on_train=True, lower_is_better=False),],
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )
    # RHCModule.register_rhc_training_step()
    SAModule.register_sa_training_step()
    # GAModule.register_ga_training_step()
    model.set_params(train_split=False, verbose=0)

    best_model = None

    X, y = train_set[:]
    X = X.numpy()
    y = y.numpy()

    gs = GridSearchCV(model, param_grid=p_grid, cv=n_splits,
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


def train_linear_svm_drybean(n_splits=3):
    """
    """
    dataset = DryBeanDataset()
    train_set, test_set = random_split(dataset, [0.8, 0.2])

    model = SVC(kernel='linear', C=0.8, gamma='scale', class_weight='balanced')
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3)

    best_model = None
    eval_accuracies = []

    X, y = train_set[:]
    X = X.numpy()
    y = y.numpy()

    with tqdm(total=skf.get_n_splits(X, y)) as pbar:
        temp_val_accs = []
        for _, (train_index, test_index) in enumerate(skf.split(X, y)):
            x_train, y_train = X[train_index], y[train_index]
            model.fit(x_train, y_train)

            x_val, y_val = X[test_index], y[test_index]
            pred = model.predict(x_val)
            acc = calculate_accuracy(pred, y_val)
            temp_val_accs.append(acc)

            if len(eval_accuracies) == 0 or np.mean(temp_val_accs) > eval_accuracies[-1]:
                best_model = copy.deepcopy(model)

            eval_accuracies.append(np.mean(temp_val_accs))
            pbar.set_postfix(val_acc=eval_accuracies[-1])
            pbar.update()

    X, y = test_set[:]
    X = X.numpy()
    y = y.numpy()
    pred = best_model.predict(X)

    acc = calculate_accuracy(pred, y)
    return best_model, eval_accuracies, acc


def train_svm_adult(kernel='linear', p_grid={}, n_splits=3, test_size=0.3, n_jobs=-1, verbose=0):
    """
    """
    n_epochs = 1

    dataset = AdultDataset()
    train_set, test_set = random_split(dataset, [0.8, 0.2])

    if kernel == 'linear':
        model = LinearSVC(dual='auto')
    else:
        model = SVC(kernel=kernel)
    best_model = None

    X, y = train_set[:]
    X = X.numpy()
    y = y.numpy()

    # Arrays to store scores
    non_nested_scores = []
    nested_scores = []

    for i in tqdm(range(n_epochs)):
        inner_cv = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=i)
        outer_cv = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=i)

        # Non_nested parameter search and scoring
        clf = HalvingGridSearchCV(estimator=model, param_grid=p_grid, cv=outer_cv, min_resources='smallest',
                                  scoring='accuracy', n_jobs=n_jobs, verbose=verbose)
        clf.fit(X, y)
        non_nested_scores.append(clf.best_score_)

        # Nested CV with parameter optimization
        nested_clf = HalvingGridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv, min_resources='smallest',
                                         scoring='accuracy', n_jobs=n_jobs, verbose=verbose)
        nested_score = cross_val_score(
            nested_clf, X=X, y=y, cv=outer_cv, n_jobs=n_jobs, verbose=verbose)

        if len(nested_scores) == 0 or nested_score.mean() > nested_scores[-1]:
            print(clf.best_params_, clf.best_score_)
            best_model = copy.deepcopy(clf.best_estimator_)

        nested_scores.append(nested_score.mean())

    X, y = test_set[:]
    X = X.numpy()
    y = y.numpy()
    pred = best_model.predict(X)

    acc = calculate_accuracy(pred, y)
    return best_model, nested_scores, acc


def set_seed(seed=123456789):
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)
    # This environment variable is required for `use_deterministic_algorithms`
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    use_deterministic_algorithms(True)


def main():
    set_seed()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # best_model, eval_accuracies, test_accuracy = train_linear_svm_drybean()
    best_model, test_accuracy = train_svm_drybean(kernel='linear', p_grid={'lr': np.logspace(-2, 1, 5),
                                                                           # 'optimizer__momentum': np.logspace(-4, 0, 5),
                                                                           'module__step_size': np.logspace(-4, 2, 10),
                                                                           #    'module__hidden_units': [10, 25, 50],
                                                                           #    'module__dropout_percent': [0., 0.1, 0.2],
                                                                           'max_epochs': [50],
                                                                           'batch_size': [16, 32, 64]}, verbose=2)
    print(f'Final test accuracy for SVM with linear kernel {test_accuracy}')
    # with open(os.path.join('checkpoints', 'drybean_svm_linear_model.pkl'), 'wb') as f:
    #     pickle.dump(best_model, f, protocol=5)
    # plot_training_curves(eval_accuracies, plot_name=os.path.join('checkpoints', 'drybean_svm_linear_acc_curves.png'))

    # best_model, eval_accuracies, test_accuracy = train_svm_drybean(kernel='rbf', p_grid={'C' : np.logspace(3, 4, 10), 'gamma' : np.logspace(-6, -4, 10)})
    # print(f'Final test accuracy for SVM with rbf kernel {test_accuracy}')
    # with open(os.path.join('checkpoints', 'drybean_svm_rbf_model.pkl'), 'wb') as f:
    #     pickle.dump(best_model, f, protocol=5)
    # plot_training_curves(eval_accuracies, plot_name=os.path.join('checkpoints', 'drybean_svm_rbf_acc_curves.png'))

    # best_model, eval_accuracies, test_accuracy = train_svm_adult(kernel='linear', p_grid={'class_weight' : [None, 'balanced'],
    #                                                                                       'dual' : ['auto'],
    #                                                                                       'multi_class' : ['ovr', 'crammer_singer'],
    #                                                                                       'max_iter' : [1000000],
    #                                                                                       'C' : np.logspace(-2, 3, 10)}, verbose=2)
    # print(f'Final test accuracy for SVM with linear kernel {test_accuracy}')
    # with open(os.path.join('checkpoints', 'adult_svm_linear_model.pkl'), 'wb') as f:
    #     pickle.dump(best_model, f, protocol=5)
    # plot_training_curves(eval_accuracies, plot_name=os.path.join('checkpoints', 'adult_svm_linear_acc_curves.png'))

    # best_model, eval_accuracies, test_accuracy = train_svm_adult(
    #     kernel='rbf', p_grid={'C': np.logspace(3, 4, 10), 'gamma': np.logspace(-6, -4, 10)})
    # print(f'Final test accuracy for SVM with rbf kernel {test_accuracy}')
    # with open(os.path.join('checkpoints', 'adult_svm_rbf_model.pkl'), 'wb') as f:
    #     pickle.dump(best_model, f, protocol=5)
    # plot_training_curves(eval_accuracies, plot_name=os.path.join(
    #     'checkpoints', 'adult_svm_rbf_acc_curves.png'))


if __name__ == '__main__':
    main()
