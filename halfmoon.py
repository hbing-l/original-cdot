import numpy as np
import pandas as pd
from da_ot import OTGroupLassoDAClassifier, OTBFBDAClassifier
from data import load_seq_two_moon_data
from plot import plot_continuous_domain_adaptation, plot_accuracies
import pdb
from sklearn.neighbors import KNeighborsClassifier



def test_cdot_methods(methods, time_reg_vector, n_samples_source=150, n_samples_targets=150, n_samples_test=1000,
                      time_length=10, max_angle=90, noise=0.1, methods_names=None, cost="seq",
                      fig_name=None, plot_mapping=True):
    Xs, ys, Xt, yt, angles = load_seq_two_moon_data(n_samples_source, n_samples_targets, time_length,
                                                    max_angle=max_angle, noise=noise)
    _, _, Xtest, ytest, _ = load_seq_two_moon_data(0, n_samples_test, time_length,
                                                       max_angle=max_angle, noise=noise)


    mapped_samples = []

    scores = np.zeros([len(methods), time_length])

    for m, da_clf in enumerate(methods):
        print("Running ", methods_names[m])
        temp_samples = []
        for k in range(time_length):
            if k > 0:
                if cost == "seq":
                    da_clf.fit(temp_samples[-1], ys, Xt[k], treg=time_reg_vector[m], Gamma_old=da_clf.Gamma,
                               Xt_old=Xt[k - 1])
                else:
                    da_clf.fit(Xs, ys, Xt[k], treg=time_reg_vector[m], Gamma_old=da_clf.Gamma,
                               Xt_old=Xt[k - 1])
            else:
                da_clf.fit(Xs, ys, Xt[k])

            temp_samples.append(da_clf.adapt_source_to_target())
            scores[m, k] = da_clf.score(Xtest[k], ytest[k])

        mapped_samples.append(temp_samples)
        # if plot_mapping:
        #     for k in range(time_length):
        #         plot_continuous_domain_adaptation(Xt, yt, mapped_samples, ys, methods_names=methods_names, time_idx=k,
        #                                           fig_name=fig_name + "_time_" + str(
        #                                               k) + "_geom.png" if fig_name is not None else None)

    # plot_accuracies(angles, scores, methods_names=methods_names,
    #                 fig_name=fig_name + "_accuracy.png" if fig_name is not None else None)

    return scores, angles


if __name__ == '__main__':
    N_RUNS = 10
    data = []
    clf = KNeighborsClassifier(n_neighbors=1)

    ot_group_lasso = OTGroupLassoDAClassifier(clf, reg=0.5, eta=10)
    ot_group_lasso_treg = OTGroupLassoDAClassifier(clf, reg=0.5, eta=10)
    ot_BFB = OTBFBDAClassifier(clf, reg=0.5, regnorm=None, it=50, epochs=1000, lr=10, verbose=True)

    # methods = [ot_group_lasso, ot_group_lasso_treg, ot_BFB]
    # methods_names =  ['ot_group_lasso', 'ot_group_lasso_treg', 'ot_BFB']
    # time_reg_vector = [0, 50, 50]

    methods = [ot_BFB]
    methods_names =  ['ot_BFB']
    time_reg_vector = [50]
    time = 10

    cost = ["seq", "direct"]
    final_scores = np.zeros([len(cost), time])
    final_losses = np.zeros([len(cost), time])

    final_scores_var = np.zeros([len(cost), time])
    final_losses_var = np.zeros([len(cost), time])

    for i, c in enumerate(cost):
        run_scores = []
        for run in range(N_RUNS):
            print("RUN %d..." % run)
            scores, angles = test_cdot_methods(
                methods=methods,
                methods_names=methods_names,
                time_reg_vector=time_reg_vector,
                fig_name="seq_run_" + str(run),
                time_length=time,
                n_samples_source=500,
                n_samples_targets=50,
                n_samples_test=1000,
                max_angle=180,
                plot_mapping=False,
                noise=0.1,
                cost=cost,
            )
            run_scores.append(scores)
            print(scores)
            print('--------------')
        
        
        avg_losses = np.mean(np.array(run_scores), axis=0)
        loss_var = np.var(np.array(run_scores), axis=0)

        final_losses[i, :] = avg_losses
        final_losses_var[i, :] = loss_var

    data.append({'mean_direct': final_losses[1, :], 'mean_order': final_losses[0, :], \
                'var_direct': final_losses_var[1, :], 'var_order': final_losses_var[0, :]})

    dataframe = pd.DataFrame(data)
    dataframe.to_csv("half_sametarget_fix_t_domain_ssh_{}.csv".format(time), index=False, sep=',')    


    # plot_accuracies(angles, avg_scores, methods_names=methods_names, fig_name="seq_avg_accuracy.png")