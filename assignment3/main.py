from torch.functional import Tensor
import torch.utils
from datasets.uciml import AdultDataset, DryBeanDataset, DatasetWrapper
from models.mlp import TwoLayerMLP
from tqdm import tqdm
import numpy as np
import copy
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import random

import itertools
from scipy import linalg
from scipy.stats import kurtosis, kurtosistest
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error, median_absolute_error
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.model_selection import KFold

import main_nn
import time


def set_seed(seed=123456789):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # This environment variable is required for `use_deterministic_algorithms`
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def plot_results(scores, title, plot_name='', y_labels=[], log_scale=False, plot_max_drop=False):
    fig, axes = plt.subplots(
        1, len(y_labels), sharex=True, sharey=False, layout='constrained')
    fig.supxlabel('Number of Components')

    std_colors = itertools.cycle(['cyan', 'magenta', 'indigo'])
    xticks = None
    for name, item in scores.items():
        if name == 'Original':
            continue
        else:
            x = np.asarray(item)[:, 0].astype(int)
            y = np.asarray(item)[:, 1]
            y_std = np.asarray(item)[:, 2]

            if name == 'PCA':
                i = 0
                axes[i].set_title('PCA')
                axes[i].plot(x, np.cumsum(y))
                axes[i].axhline(y=1.0, color='r', linestyle='--')
            elif name == 'ICA':
                i = 1
                axes[i].set_title('ICA')
                axes[i].axhline(y=3.0, color='r', linestyle='--')
                axes[i].set_ylim([-1, 10])
            else:
                i = 2
                axes[i].set_title('Random Projection')
                # axes[i].axhline(y=1e-3, color='r', linestyle='--')
                max_drop = np.argmin(y[1:] - y[:-1]) + 1
                axes[i].plot(x[max_drop], y[max_drop], 'rx')
            if plot_max_drop:
                max_drop = np.argmin(y[1:] - y[:-1]) + 1
                plt.plot(x[max_drop], y[max_drop], 'go')
            if xticks is None:
                xticks = x

            if i != 0:
                axes[i].plot(x, y, label=name)
            # axes[i].fill_between(x, y - y_std, y + y_std,
            #                      alpha=0.3, color=next(std_colors))
            axes[i].set_ylabel(y_labels[i], fontsize='small')
            axes[i].grid(visible=True)
            axes[i].set_xticks(x)
            if i == 2:
                axes[i].legend()

    if 'Original' in scores:
        plt.axhline(scores['Original'], linestyle='--',
                    color='red', label='Original')
    # plt.xticks(xticks)
    # if log_scale:
    #     plt.yscale('log')
    # plt.grid(visible=True)
    # plt.legend()
    # plt.title(title)
    # plt.autoscale()
    # plt.xlabel('Number of Clusters')
    plt.savefig(plot_name)
    plt.close()


def plot_reconstruction_error(scores, title, plot_name='', log_scale=True):
    std_colors = itertools.cycle(['cyan', 'magenta', 'indigo'])
    xticks = None
    for name, item in scores.items():
        if name == 'Original':
            continue
        else:
            x = np.asarray(item)[:, 0].astype(int)
            y = np.asarray(item)[:, 1]
            y_std = np.asarray(item)[:, 2]
            if xticks is None:
                xticks = x
            plt.plot(x, y, label=name)
            plt.fill_between(x, y - y_std, y + y_std,
                             alpha=0.3, color=next(std_colors))
    if 'Original' in scores:
        plt.axhline(scores['Original'], linestyle='--',
                    color='red', label='Original')
    plt.xticks(xticks)
    if log_scale:
        plt.yscale('log')
    plt.grid(visible=True)
    plt.legend()
    plt.title(title)
    plt.autoscale()
    plt.xlabel('Number of Clusters')
    plt.savefig(plot_name)
    plt.close()


def plot_silhouette(X, cluster_labels, original_score, n_clusters, ax):
    ax.set_xlim([-1, 1])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    silhouette_avg = silhouette_score(X, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_ylabel(f"n = {n_clusters}")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    # The vertical line for the silhouette score of the original dataset
    if original_score is not None:
        ax.axvline(x=original_score, color="blue", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    # ax.set_xticks([-1, -0.5, 0., 0.5, 1])


def plot_learning_curves(results, plot_name='loss_curves.png'):
    """
    Plot the training loss and accuracy curves
    """
    for name, (_, _, _, eval_accuracies, _, _) in results.items():
        if name == 'Original':
            plt.plot(eval_accuracies, 'r--', label=name)
        else:
            plt.plot(eval_accuracies, label=name)
    plt.legend()
    plt.grid(visible=True, markevery=1)
    plt.xlabel('Epoch')
    plt.title('Learning Curve')
    plt.savefig(plot_name)
    plt.close()


def plot_learning_curves_2(results, y_label='', plot_name='loss_curves.png'):
    """
    Plot the training loss and accuracy curves
    """
    width = 0.1
    multiplier = 0
    n_keys = len(list(results.keys()))

    _, ax = plt.subplots(layout='constrained')
    for name, item in results.items():
        temp = dict(item)
        x = np.array(list(temp.keys()))
        accs = np.array([temp[k] for k in x])

        if name == 'Original':
            plt.axhline(accs[0], color="red", linestyle="--")
            continue

        offset = width * multiplier
        rects = ax.bar(x + offset - ((n_keys // 2) * width),
                       accs, width, label=name)
        ax.bar_label(rects, padding=1, fmt='')
        ax.set_xticks(x)
        multiplier += 1
    plt.ylim(0.2, 1.0)
    plt.grid(visible=True, axis='y')
    plt.legend(loc='lower right', ncol=n_keys)
    plt.xlabel(y_label)
    plt.ylabel('Accuracy')
    plt.savefig(plot_name)
    plt.close()


def get_score(X, y_pred, random_state=0):
    return silhouette_score(X, y_pred, random_state=random_state)


def apply_clustering_and_predict(X, n_components=3, random_state=0):
    return KMeans(n_clusters=n_components, random_state=random_state).fit_predict(X)


def plot_data_histogram(datasets, dataset_names):
    for dataset, dataset_name in zip(datasets, dataset_names):
        _, axes = plt.subplots(1, 2, sharex=False, layout='constrained')

        df = dataset.get_dataframe()
        # if dataset_name == 'adult':
        #     df = df.drop(columns=[2, 4, 10, 11, 0, 12])
        # else:
        #     df = df.drop(columns=[0, 1, 2, 3, 6, 7])
        print(df.describe())
        df.plot.hist(bins=50, alpha=0.3, ax=axes[1])
        # axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_xlabel('Value')

        axes[0].bar(range(dataset.get_num_features()), df.mean(
                axis=0, skipna=True), yerr=df.std(axis=0))
        axes[0].set_xticks([])
        axes[0].set_ylabel('Value')
        axes[0].set_xlabel('Feature')
        axes[0].set_yscale('log')

        # ax2 = axes[0].twinx()
        # X, y = dataset[:]
        # X = X.numpy()
        # score = kurtosis(X, axis=0, fisher=False)
        # ax2.plot(range(dataset.get_num_features()), score, color='r', linestyle='--')
        # ax2.axhline(np.mean(score), color='g', linestyle='-.')
        # ax2.set_ylabel('Kurtosis')
        # ax2.set_ylim([-1, 20])
        # ax2.grid(visible=True)

        # ax2.set_yticks([])
        # axes[0].set_ylabel('Value')
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.xticks([])
        # plt.ylabel('Value')

        
        # ax2.set_yscale('log')
        # ax2.grid(visible=True)

        # df.plot(kind='bar', title='Feature scaling', logy=True, y=df.mean(axis=0), yerr=df.std(axis=0))
        # plt.xlim(right=20)
        # plt.xlabel('Values')
        # plt.title('Features histogram')
        plt.savefig(f"{dataset_name}_features.png")
        plt.close()


def plot_data_histogram_2(datasets, dataset_names):
    for dataset, dataset_name in zip(datasets, dataset_names):
        X, _ = dataset[:]
        X = X.numpy()

        n_features = 5
        _, axes = plt.subplots(n_features, n_features, sharey=True, sharex=True)

        for i in range(n_features):
            for j in range(n_features):
                axes[i, j].scatter(X[:,i], X[:,j])
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

        plt.savefig(f"{dataset_name}_features.png")
        plt.close()

def clustering(datasets, dataset_names, use_pct=0.3):
    for dataset, dataset_name in zip(datasets, dataset_names):
        set_seed()
        use_set, _ = random_split(dataset, [use_pct, 1.0 - use_pct])
        X, y = use_set[:]
        X = X.numpy()
        y = y.numpy()

        original_score = get_score(X, y)

        n_clusters = np.arange(2, 11, dtype=int)
        cluster_algs = [
            KMeans(max_iter=300, init='random', n_init=10, random_state=0)]
        names = ['KMeans']
        for algo_name, algo in zip(names, cluster_algs):
            fig, axes = plt.subplots(
                3, 3, sharex=True, sharey=True, layout='constrained')
            fig.supxlabel('Silhouette Coeff.')

            axes = axes.flatten()
            print(f"Running clustering on {dataset_name} using {algo_name}")
            for i, n in enumerate(tqdm(n_clusters)):
                if isinstance(algo, GaussianMixture):
                    algo = algo.set_params(n_components=n)
                else:
                    algo = algo.set_params(n_clusters=n)
                y_pred = algo.fit_predict(X)
                plot_silhouette(X, y_pred, original_score, n, axes[i])

            if len(n_clusters) < len(axes):
                for ax in np.asarray(axes)[len(n_clusters):]:
                    ax.set_visible(False)
            plt.savefig(os.path.join('checkpoints',
                        f"{dataset_name}_clustering_{algo_name}.png"))
            plt.close()


def clustering_2(configs, use_pct=0.3):
    for config in configs:
        dataset, dataset_name, cluster_algs, cluster_names, n_clusters = config

        set_seed()
        use_set, _ = random_split(dataset, [use_pct, 1.0 - use_pct])
        X, y = use_set[:]
        if isinstance(X, Tensor):
            X = X.numpy()
            y = y.numpy()

        original_score = get_score(X, y)
        times = {}
        for algo_name, algo in zip(cluster_names, cluster_algs):
            fig, axes = plt.subplots(
                (n_clusters.shape[0] // 3 + 1), 3, sharex=True, sharey=True, layout='constrained')
            fig.supxlabel('Silhouette Coeff.')

            axes = axes.flatten()
            print(f"Running clustering on {dataset_name} using {algo_name}")
            for i, n in enumerate(tqdm(n_clusters)):
                if isinstance(algo, GaussianMixture):
                    algo = algo.set_params(n_components=n)
                else:
                    algo = algo.set_params(n_clusters=n)
                start = time.time()
                y_pred = algo.fit_predict(X)
                if algo_name not in times:
                    times[algo_name] = []
                times[algo_name].append(time.time() - start)
                plot_silhouette(X, y_pred, original_score, n, axes[i])

            if len(n_clusters) < len(axes):
                for ax in np.asarray(axes)[len(n_clusters):]:
                    ax.set_visible(False)
            plt.savefig(os.path.join('checkpoints',
                        f"{dataset_name}_clustering_{algo_name}.png"))
            plt.close()
        
        for name, item in times.items():
            plt.plot(n_clusters, item, label=name)
        plt.xlabel('Number of clusters')
        plt.ylabel('Execution time (sec)')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join('checkpoints',
                    f"{dataset_name}_clustering_exec_time.png"))
        plt.close()


def clustering_mog(datasets, dataset_names, use_pct=0.3):
    for dataset, dataset_name in zip(datasets, dataset_names):
        set_seed()
        use_set, _ = random_split(dataset, [use_pct, 1.0 - use_pct])
        X, y = use_set[:]
        X = X.numpy()
        y = y.numpy()

        n_clusters = np.arange(2, 11, dtype=int)
        cluster_algs = [GaussianMixture(
            max_iter=200, init_params='random', n_init=10, random_state=0)]
        names = ['Mixture of Gaussian']
        for algo_name, algo in zip(names, cluster_algs):
            print(f"Running clustering on {dataset_name} using {algo_name}")

            scores_bic = []
            scores_aic = []
            for _, n in enumerate(tqdm(n_clusters)):
                algo = algo.set_params(n_components=n)
                algo.fit(X)
                scores_bic.append(algo.bic(X))
                scores_aic.append(algo.aic(X))

            scores_bic = np.asarray(scores_bic)
            scores_aic = np.asarray(scores_aic)
            plt.plot(n_clusters, scores_bic, label='BIC')
            # plt.plot(n_clusters, scores_aic, label='AIC')
            max_drop_bic = np.argmin(scores_bic[1:] - scores_bic[:-1]) + 1
            plt.plot(n_clusters[max_drop_bic], scores_bic[max_drop_bic], 'go')
            plt.xlabel('Number of Clusters')
            plt.ylabel('BIC criterion')
            plt.grid(visible=True)
            plt.savefig(os.path.join('checkpoints',
                        f"{dataset_name}_clustering_{algo_name}.png"))
            plt.close()


def dimension_reduction(datasets, dataset_names, n_clusters, use_pct=0.3, use_reconstruction_error=False):
    for dataset, dataset_name, n_cluster_range in zip(datasets, dataset_names, n_clusters):
        scores = {}

        set_seed()
        use_set, _ = random_split(dataset, [use_pct, 1.0 - use_pct])
        X, y = use_set[:]
        X = X.numpy()
        y = y.numpy()

        data = KFold(n_splits=5, shuffle=True)

        dr_algs = [PCA(random_state=0),
                   FastICA(max_iter=20000, tol=1e-3,
                           whiten_solver='svd', random_state=0),
                   SparseRandomProjection(random_state=0),
                   GaussianRandomProjection(random_state=0)]
        names = ['PCA', 'ICA', 'PRP', 'GRP']
        for algo_name, algo in zip(names, dr_algs):
            configs = []
            for n in tqdm(n_cluster_range):
                algo = algo.set_params(n_components=n)

                temp_scores = []
                temp_times = []
                for train_idx, eval_idx in data.split(X, y):
                    start = time.time()
                    algo = algo.fit(X[train_idx])
                    transformed_X = algo.transform(X[train_idx])
                    configs.append(
                        (DatasetWrapper(transformed_X, y[train_idx], transformed_X.shape[1]), f"{dataset_name}_{algo_name}_{n}_transformed",
                         [GaussianMixture(max_iter=200, init_params='kmeans', n_init=10, random_state=0),
                          KMeans(max_iter=300, init='random', n_init=10, random_state=0)],
                         ['MoG', 'KMeans'], np.arange(2, n + 2, dtype=int)),
                    )
                    temp_times.append(time.time() - start)
                    if algo_name == 'PCA':
                        if use_reconstruction_error:
                            transformed_X = algo.transform(X[eval_idx])
                            reconstructed_X = algo.inverse_transform(
                                transformed_X)
                            temp_scores.append(
                                mean_squared_error(X[eval_idx], reconstructed_X))
                        else:
                            temp_scores.append(algo.explained_variance_ratio_)
                    elif algo_name == 'ICA':
                        if use_reconstruction_error:
                            transformed_X = algo.transform(X[eval_idx])
                            reconstructed_X = algo.inverse_transform(
                                transformed_X)
                            temp_scores.append(
                                mean_squared_error(X[eval_idx], reconstructed_X))
                        else:
                            transformed_X = algo.transform(X[eval_idx])
                            temp_scores.append(
                                np.mean(kurtosis(transformed_X, axis=0, fisher=False)))
                    else:
                        transformed_X = algo.transform(X[eval_idx])
                        reconstructed_X = algo.inverse_transform(transformed_X)
                        temp_scores.append(
                            mean_squared_error(X[eval_idx], reconstructed_X))
                if algo_name not in scores:
                    scores[algo_name] = []
                scores[algo_name].append(
                    [n, np.mean(temp_scores), np.std(temp_scores), np.mean(temp_times)])
            clustering_2(configs, use_pct=1.0)

        for name, item in scores.items():
            x = np.asarray(item)[:, 0].astype(int)
            y = np.asarray(item)[:, 3]
            plt.plot(x, y, label=name)
        plt.legend()
        plt.yscale('log')
        plt.ylabel('Execution time (sec)')
        plt.xlabel('Number of components')
        plt.grid(visible=True)
        plt.savefig(os.path.join(
                    'checkpoints', f"{dataset_name}_dr_exec_time.png"))
        plt.close()

        if use_reconstruction_error:
            plot_reconstruction_error(scores, 'Dimension Reduction', os.path.join(
                'checkpoints', f"{dataset_name}_dimension_reduction_reconstruction.png"))
        else:
            plot_results(scores, 'Dimension Reduction', os.path.join(
                'checkpoints', f"{dataset_name}_dimension_reduction.png"), y_labels=['% Explained Variance', 'Kurtosis', 'MSE'])


def dimension_reduction_then_clustering(configs, use_pct=0.3):
    for config in configs:
        dataset, dataset_name, dr_algs, dr_names, cluster_algs, cluster_names = config

        set_seed()
        use_set, _ = random_split(dataset, [use_pct, 1.0 - use_pct])
        X, y = use_set[:]
        X = X.numpy()
        y = y.numpy()

        original_score = get_score(X, y)

        fig, axes = plt.subplots(
            len(dr_names), len(cluster_names), sharex=True, sharey=True, layout='constrained')
        fig.supxlabel('Silhouette Coeff.')

        i = 0
        with tqdm(total=len(dr_names) * len(cluster_names)) as pbar:
            for algo_name, algo in zip(dr_names, dr_algs):
                n_components = algo.get_params()['n_components']
                transformed_X = algo.fit_transform(X)
                j = 0
                for cluster_algo_name, cluster_algo in zip(cluster_names, cluster_algs):
                    n_clusters = transformed_X.shape[1]
                    best_score = None
                    best_n = None
                    for n in range(2, n_clusters+1):
                        if isinstance(cluster_algo, GaussianMixture):
                            cluster_algo.set_params(n_components=n)
                        else:
                            cluster_algo.set_params(n_clusters=n)
                        y_pred = cluster_algo.fit_predict(transformed_X)
                        score = silhouette_score(transformed_X, y_pred)
                        if best_score is None or score > best_score:
                            best_score = score
                            best_n = n
                    if isinstance(cluster_algo, GaussianMixture):
                        cluster_algo.set_params(n_components=best_n)
                    else:
                        cluster_algo.set_params(n_clusters=best_n)
                    y_pred = cluster_algo.fit_predict(transformed_X)
                    plot_silhouette(transformed_X, y_pred,
                                    original_score, best_n, axes[i, j])
                    axes[i, j].set_ylabel(
                        f"{algo_name}({n_components}) +\n {cluster_algo_name}({best_n})", fontsize='small')
                    j += 1
                    pbar.update()
                i += 1
        plt.savefig(os.path.join('checkpoints',
                    f"{dataset_name}_dr_clustering.png"))
        plt.close()


def dimension_reduction_then_mlp(datasets, dataset_names, use_pct=0.3, n_epochs=50):
    results = {}
    for dataset, dataset_name in zip(datasets, dataset_names):
        set_seed()

        use_set, _ = random_split(dataset, [use_pct, 1.0 - use_pct])
        X, y = use_set[:]
        X = X.numpy()
        y = y.numpy()

        if dataset_name == 'adult':
            _, _, _, _, _, _, original_final_acc = main_nn.train_linear_adult(
                DatasetWrapper(X, y, dataset.get_num_classes()), n_epochs=n_epochs)
        else:
            _, _, _, _, _, _, original_final_acc = main_nn.train_linear_drybean(
                DatasetWrapper(X, y, dataset.get_num_classes()), n_epochs=n_epochs)
        results['Original'] = []

        dr_algs = [PCA(random_state=0),
                   FastICA(max_iter=20000, tol=1e-3,
                           whiten_solver='svd', random_state=0),
                   SparseRandomProjection(random_state=0),
                   GaussianRandomProjection(random_state=0)]
        dr_names = ['PCA', 'ICA', 'PRP', 'GRP']

        n_components = np.arange(1, dataset.get_num_features(), dtype=int)
        for algo_name, algo in zip(dr_names, dr_algs):
            for n_component in n_components:
                algo.set_params(n_components=n_component)
                ds = DatasetWrapper(algo.fit_transform(
                    X), y, dataset.get_num_classes())
                if dataset_name == 'adult':
                    _, _, _, _, _, _, final_acc = main_nn.train_linear_adult(
                        ds, n_epochs=n_epochs)
                else:
                    _, _, _, _, _, _, final_acc = main_nn.train_linear_drybean(
                        ds, n_epochs=n_epochs)
                if algo_name not in results:
                    results[algo_name] = []
                results[algo_name].append((n_component, final_acc))
                results['Original'].append((n_component, original_final_acc))
        plot_learning_curves_2(results, y_label='Number of components', plot_name=os.path.join(
            'checkpoints', f"{dataset_name}_dr_mlp_learning_curves.png"))


def clustering_then_mlp(datasets, dataset_names, use_pct=0.3, n_epochs=50):
    results = {}
    for dataset, dataset_name in zip(datasets, dataset_names):
        set_seed()

        use_set, _ = random_split(dataset, [use_pct, 1.0 - use_pct])
        X, y = use_set[:]
        X = X.numpy()
        y = y.numpy()

        if dataset_name == 'adult':
            _, _, _, _, _, _, original_final_acc = main_nn.train_linear_adult(
                DatasetWrapper(X, y, dataset.get_num_classes()), n_epochs=n_epochs)
        else:
            _, _, _, _, _, _, original_final_acc = main_nn.train_linear_drybean(
                DatasetWrapper(X, y, dataset.get_num_classes()), n_epochs=n_epochs)
        results['Original'] = []

        cluster_algs = [GaussianMixture(max_iter=200, init_params='random', n_init=10, random_state=0),
                        KMeans(max_iter=300, init='random', n_init=10, random_state=0)]
        cluster_names = ['MoG', 'KMeans']

        n_components = np.arange(2, dataset.get_num_features(), dtype=int)
        for algo_name, algo in zip(cluster_names, cluster_algs):
            for n_component in n_components:
                if isinstance(algo, KMeans):
                    algo.set_params(n_clusters=n_component)
                    ds = DatasetWrapper(algo.fit_transform(
                        X), y, dataset.get_num_classes())
                else:
                    algo.set_params(n_components=n_component)
                    ds = DatasetWrapper(algo.fit(X).predict_proba(
                        X), y, dataset.get_num_classes())
                if dataset_name == 'adult':
                    _, _, _, _, _, _, final_acc = main_nn.train_linear_adult(
                        ds, n_epochs=n_epochs)
                else:
                    _, _, _, _, _, _, final_acc = main_nn.train_linear_drybean(
                        ds, n_epochs=n_epochs)
                if algo_name not in results:
                    results[algo_name] = []
                results[algo_name].append((n_component, final_acc))
                results['Original'].append((n_component, original_final_acc))
        plot_learning_curves_2(results, y_label='Number of clusters', plot_name=os.path.join(
            'checkpoints', f"{dataset_name}_cluster_mlp_learning_curves.png"))


def main():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    def log(X):
        return np.log(np.clip(X, a_min=1e-9, a_max=None))

    def standardize(X):
        return (X - X.mean(dim=0)) / X.std(dim=0)

    def normalize(X):
        return (X - X.min(dim=0)[0]) / (X.max(dim=0)[0] - X.min(dim=0)[0])

    datasets = [DryBeanDataset(transforms=None),
                AdultDataset(transforms=None)]
    dataset_names = ['drybean', 'adult']
    n_clusters = [np.arange(1, datasets[0].get_num_features(), dtype=int),
                  np.arange(1, datasets[1].get_num_features(), dtype=int)]

    plot_data_histogram(datasets, dataset_names)

    configs = [
        # Drybean dataset
        (DryBeanDataset(transforms=standardize), 'drybean',
         [GaussianMixture(max_iter=200, init_params='kmeans', n_init=10, random_state=0),
          KMeans(max_iter=300, init='random', n_init=10, random_state=0)],
         ['MoG', 'KMeans'], np.arange(2, datasets[0].get_num_features(), dtype=int)),
        # Adult dataset
        (AdultDataset(transforms=standardize), 'adult',
         [GaussianMixture(max_iter=200, init_params='kmeans', n_init=10, random_state=0),
          KMeans(max_iter=300, init='random', n_init=10, random_state=0)],
         ['MoG', 'KMeans'], np.arange(2, datasets[1].get_num_features(), dtype=int))
    ]
    print("Running clustering")
    clustering_2(configs)

    print("Running dimension reduction")
    dimension_reduction(datasets, dataset_names, n_clusters, use_reconstruction_error=False, use_pct=0.1)

    print("Running dimension reduction then clustering")
    configs = [
        # Drybean dataset
        (DryBeanDataset(transforms=normalize), 'drybean',
         [PCA(n_components=2, random_state=0),
          FastICA(n_components=2, max_iter=20000, tol=1e-3,
                  whiten_solver='svd', random_state=0),
          SparseRandomProjection(n_components=3, random_state=0),
          GaussianRandomProjection(n_components=4, random_state=0)],
         ['PCA', 'ICA', 'PRP', 'GRP'],
         [GaussianMixture(max_iter=200, init_params='random', n_init=10, random_state=0),
          KMeans(max_iter=300, init='random', n_init=10, random_state=0)],
         ['MoG', 'KMeans']),
        # Adult dataset
        (AdultDataset(transforms=normalize), 'adult',
         [PCA(n_components=4, random_state=0),
          FastICA(n_components=5, max_iter=20000, tol=1e-3,
                  whiten_solver='svd', random_state=0),
          SparseRandomProjection(n_components=4, random_state=0),
          GaussianRandomProjection(n_components=5, random_state=0)],
         ['PCA', 'ICA', 'PRP', 'GRP'],
         [GaussianMixture(max_iter=200, init_params='random', n_init=10, random_state=0),
          KMeans(max_iter=300, init='random', n_init=10, random_state=0)],
         ['MoG', 'KMeans'])
    ]
    dimension_reduction_then_clustering(configs)

    print("Running dimension reduction then train MLP")
    dimension_reduction_then_mlp(datasets, dataset_names, use_pct=0.05)

    print("Running clustering then train MLP")
    clustering_then_mlp(datasets, dataset_names)


if __name__ == '__main__':
    main()
