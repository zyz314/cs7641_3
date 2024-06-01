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
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, HalvingGridSearchCV, cross_val_score, StratifiedShuffleSplit

def plot_training_curves(eval_accuracies, plot_name = 'loss_curves.png'):
    """
    Plot the training loss and accuracy curves
    """
    plt.plot(range(len(eval_accuracies)), eval_accuracies, label='eval_acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(plot_name)


def calculate_accuracy(pred : np.ndarray, target : np.ndarray) -> float:
    """
    Calculate accuracy of prediction
    """
    return (np.sum(pred == target) / target.shape[0]).item()


def train_svm_drybean(kernel='linear'):
    """
    """
    n_epochs = 2

    dataset = DryBeanDataset()
    train_set, test_set = random_split(dataset, [0.8, 0.2])

    model = SVC(kernel=kernel)
    best_model = None

    X, y = train_set[:]
    X = X.numpy()
    y = y.numpy()

    p_grid = {'C' : np.logspace(3, 4, 10), 'gamma' : np.logspace(-6, -4, 10)}

    # Arrays to store scores
    non_nested_scores = []
    nested_scores = []

    for i in tqdm(range(n_epochs)):
        inner_cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=i)
        outer_cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=i)

        # Non_nested parameter search and scoring
        clf = HalvingGridSearchCV(estimator=model, param_grid=p_grid, cv=outer_cv, scoring='accuracy', refit=True, n_jobs=3, verbose=2)
        clf.fit(X, y)
        non_nested_scores.append(clf.best_score_)

        # Nested CV with parameter optimization
        nested_clf = HalvingGridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv, scoring='accuracy', refit=True, n_jobs=3, verbose=2)
        nested_score = cross_val_score(nested_clf, X=X, y=y, cv=outer_cv)

        if len(nested_scores) == 0 or nested_score.mean() > nested_scores[-1]:
            print(clf.best_params_)
            best_model = copy.deepcopy(clf.best_estimator_)

        nested_scores.append(nested_score.mean())

    X, y = test_set[:]
    X = X.numpy()
    y = y.numpy()
    pred = best_model.predict(X)

    acc = calculate_accuracy(pred, y)
    return best_model, nested_scores, acc


def set_seed(seed = 123456789):
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

    # best_model, eval_accuracies, test_accuracy = train_svm_drybean(kernel='linear', class_weight='balanced')
    # print(f'Final test accuracy for SVM with linear kernel {test_accuracy}')    
    # with open(os.path.join('checkpoints', 'drybean_svm_linear_model.pkl'), 'wb') as f:
    #     pickle.dump(best_model, f, protocol=5)
    # plot_training_curves(eval_accuracies, plot_name=os.path.join('checkpoints', 'drybean_svm_linear_acc_curves.png'))

    best_model, eval_accuracies, test_accuracy = train_svm_drybean(kernel='rbf')
    print(f'Final test accuracy for SVM with rbf kernel {test_accuracy}')
    with open(os.path.join('checkpoints', 'drybean_svm_rbf_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f, protocol=5)
    plot_training_curves(eval_accuracies, plot_name=os.path.join('checkpoints', 'drybean_svm_rbf_acc_curves.png'))


if __name__ == '__main__':
    main()
