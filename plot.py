import numpy as np

import matplotlib.pyplot as plt


def plot_continuous_domain_adaptation(X, y, X_mapped, y_mapped, time_idx=0, methods_names=None, fig_id=None,
                                      fig_name=None):
    n_methods = len(X_mapped)

    plt.figure(fig_id, figsize=(5 * n_methods, 5))
    for i in range(n_methods):
        plt.subplot(1, n_methods, i + 1, aspect='equal')
        plt.scatter(X[time_idx][:, 0], X[time_idx][:, 1], c=y[time_idx], marker='o',
                    label='Target samples', alpha=0.5)
        plt.scatter(X_mapped[i][time_idx][:, 0], X_mapped[i][time_idx][:, 1], c=y_mapped,
                    marker='+', label='Transp samples', s=30)
        if methods_names is not None:
            plt.title(methods_names[i] + ': Source to Target')
        else:
            plt.title('Source to Target')
        plt.xticks([])
        plt.yticks([])

    if fig_name is not None:
        plt.savefig(fig_name)
        plt.close()
    else:
        plt.show()


def plot_accuracies(transform_values, names, scores, vars, methods_names=None, fig_id=None, fig_name=None):
    n_methods = scores.shape[0]

    plt.figure(fig_id)
    for i in range(n_methods):
        plt.plot(transform_values, np.transpose(scores[i, :]),
                 label=methods_names[i] if methods_names is not None else None)
        plt.fill_between(transform_values, np.transpose(scores[i, :] + vars[i, :]), np.transpose(scores[i, :] - vars[i, :]), alpha=0.12)
    plt.legend()
    plt.xticks(transform_values, names)

    if fig_name is not None:
        plt.savefig(fig_name)
        plt.close()
    else:
        plt.show()
