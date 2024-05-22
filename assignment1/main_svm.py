from datasets.uciml import AdultDataset, DryBeanDataset
import numpy as np
import os
import pickle

from torch import manual_seed, use_deterministic_algorithms
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import random

from sklearn.svm import LinearSVC, SVC

def plot_training_curves(train_losses, train_accuracies, eval_losses, eval_accuracies, plot_name = 'loss_curves.png'):
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


def calculate_accuracy(pred : np.ndarray, target : np.ndarray) -> float:
    """
    Calculate accuracy of prediction
    """
    return (np.sum(pred == target) / target.shape[0]).item()

def train_svm_drybean():
    """
    """
    dataset = DryBeanDataset()
    train_set, test_set = random_split(dataset, [0.8, 0.2])

    model = LinearSVC()

    X, y = train_set[:]
    X = X.numpy()
    y = y.numpy()

    model.fit(X, y)
    
    X, y = test_set[:]
    X = X.numpy()
    y = y.numpy()
    pred = model.predict(X)

    acc = calculate_accuracy(pred, y)
    print(f'Final test accuracy {np.mean(acc)}')
    return model


def set_seed(seed = 123456789):
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)
    # This environment variable is required for `use_deterministic_algorithms`
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    use_deterministic_algorithms(True)


def main():
    set_seed()

    best_model = train_svm_drybean()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    with open(os.path.join('checkpoints', 'drybean_svm_best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f, protocol=5)
    # plot_training_curves(train_losses[1:], train_accuracies[1:], eval_losses[1:], eval_accuracies[1:],
    #                      plot_name=os.path.join('checkpoints', 'drybean_loss_curves.png'))

if __name__ == '__main__':
    main()
