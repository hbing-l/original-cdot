from os import times_result
from random import random
from typing import final
import numpy as np
# from torch import R
from da_ot import OTGroupLassoDAClassifier, OTBFBDAClassifier
from data import Xt_all, load_battery_data, load_battery_data_random
from plot import plot_continuous_domain_adaptation, plot_accuracies

from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn import tree
import random


def test_cdot_methods(methods, time_reg_vector, n_samples_source, n_samples_targets,
                      time_length, time_series, methods_names=None, cost="seq",
                      fig_name=None, plot_mapping=False, random_seed = 0):
    
    # Xs, ys, Xt, yt, Xt_all, yt_all = load_battery_data(n_samples_source, n_samples_targets, time_length, True)
    Xs, ys, Xt, yt, Xt_all, yt_all = load_battery_data_random(n_samples_source, n_samples_targets, time_series, shuffle_or_not = True, random_seed = random_seed)

    mapped_samples = []

    scores = np.zeros([len(methods), time_length])
    losses = np.zeros([len(methods), time_length])
    ots = np.arange(time_length)

    for m, da_clf in enumerate(methods):
        print("Running ", methods_names[m])
        temp_samples = []
        for k in range(time_length):
            if k > 0:
                if cost == "seq":
                    da_clf.fit(temp_samples[-1], ys, Xt[k], treg=time_reg_vector[m], Gamma_old=da_clf.Gamma,
                               Xt_old=Xt[k - 1])
                    print("--seq oting--")
                else:
                    # da_clf.fit(Xs, ys, Xt[k], treg=time_reg_vector[m], Gamma_old=da_clf.Gamma,
                    #            Xt_old=Xt[k - 1])
                    da_clf.fit(Xs, ys, Xt[k])
                    print("--direct oting--")

            else:
                da_clf.fit(Xs, ys, Xt[k])
    
            temp_samples.append(da_clf.adapt_source_to_target())

            result = da_clf.predict(Xt_all[k])
            scores[m, k] = r2_score(yt_all[k], result)
            losses[m, k] = mean_squared_error(yt_all[k], result)

        mapped_samples.append(temp_samples)
        # if plot_mapping:
        #     for k in range(time_length):
        #         plot_continuous_domain_adaptation(Xt, yt, mapped_samples, ys, methods_names=methods_names, time_idx=k,
        #                                           fig_name=fig_name + "_time_" + str(
        #                                               k) + "_geom.png" if fig_name is not None else None)

    # plot_accuracies(ots, scores, methods_names=methods_names,
    #                 fig_name='./fig/' + fig_name + "_" + str(time_length) + "_accuracy.png" if fig_name is not None else None)
    # plot_accuracies(ots, losses, methods_names=methods_names,
    #                 fig_name='./fig/' + fig_name + "_" + str(time_length) + "_loss.png" if fig_name is not None else None)
  

    return scores, losses, ots


if __name__ == '__main__':
    N_RUNS = 10

    # clf = KNeighborsClassifier(n_neighbors=1)
    # clf = ensemble.RandomForestRegressor(n_estimators=4)
    # clf = Ridge(alpha=5)
    clf = svm.SVR()
    # clf = tree.DecisionTreeRegressor()


    ot_group_lasso = OTGroupLassoDAClassifier(clf, reg=0.5, eta=10)
    ot_group_lasso_treg = OTGroupLassoDAClassifier(clf, reg=0.5, eta=10)
    ot_BFB = OTBFBDAClassifier(clf, reg=0.1, regnorm=None, it=50, epochs=1000, lr=10, verbose=True)

    # methods = [ot_group_lasso, ot_group_lasso_treg, ot_BFB]
    # methods_names =  ['ot_group_lasso', 'ot_group_lasso_treg', 'ot_BFB']
    # time_reg_vector = [0, 50, 50]

    methods = [ot_BFB]
    methods_names =  ['ot_BFB']
    time_reg_vector = [50]

    for sd in range(20):
        np.random.seed(sd)
        time = np.random.randint(1, 10)
        # ot_series = 5 * np.random.randint(2, 10, time)
        ot_series = 5 * (np.random.choice(9, time, replace=False) + 2)

        sorted_ot_series = ot_series

        sorted_ot_series = sorted_ot_series[np.where(sorted_ot_series <= sorted_ot_series[-1])]
        sorted_ot_series = np.sort(sorted_ot_series)

        sample_num = len(sorted_ot_series) - 1
        unorder_sd = 0
        unsorted_ot_series = sorted_ot_series
        if sample_num > 0 and time > 1:
            random.seed(unorder_sd)
            sample_list = [i for i in range(time - 1)]
            sample_list = random.sample(sample_list, sample_num)

            unsorted_ot_series = ot_series[sample_list]
            unsorted_ot_series = np.append(unsorted_ot_series, ot_series[-1])
        
        sorted_ot_series = unsorted_ot_series

        # if time > 1:
        #     i = time - 2
        #     while True:
        #         if i < 0:
        #             break
        #         if sorted_ot_series[i] >= sorted_ot_series[i + 1]:
        #             sorted_ot_series = np.delete(sorted_ot_series, i)
        #         i -= 1


        # print(ot_series)
        # print(sorted_ot_series)
        # print(unsorted_ot_series)

        time = len(sorted_ot_series)
        target = 10
        cost = ["seq", "direct"]
        final_scores = np.zeros([len(cost), time])
        final_losses = np.zeros([len(cost), time])

        final_scores_var = np.zeros([len(cost), time])
        final_losses_var = np.zeros([len(cost), time])

        for i, c in enumerate(cost):
            print('-----{} ot-----'.format(c))

            run_scores = []
            run_losses = []

            for run in range(N_RUNS):
                print("RUN %d..." % run)
                scores, losses, ots = test_cdot_methods(
                    methods=methods,
                    methods_names=methods_names,
                    time_reg_vector=time_reg_vector,
                    fig_name="seq_run_" + str(run),
                    time_length=time,
                    time_series=sorted_ot_series,
                    n_samples_source=67,
                    n_samples_targets=target,
                    plot_mapping=False,
                    cost=c,
                    random_seed = sd * (run+1)
                )
                run_scores.append(scores)
                run_losses.append(losses)
                print('----scores----')
                print(scores)
                print('----losses----')
                print(losses)

            avg_scores = np.mean(np.array(run_scores), axis=0)
            score_var = np.var(np.array(run_scores), axis=0)
            
            avg_losses = np.mean(np.array(run_losses), axis=0)
            loss_var = np.var(np.array(run_losses), axis=0)

            final_scores[i, :] = avg_scores
            final_losses[i, :] = avg_losses

            final_scores_var[i, :] = score_var
            final_losses_var[i, :] = loss_var

        # print("ot series: ", ot_series)
        # print("sorted ot series: ", sorted_ot_series)
        # print("unsorted ot series: ", unsorted_ot_series)
        # print("average loss: ", final_losses[:, -1])
        # plot_accuracies(ots, sorted_ot_series, final_scores, final_scores_var, methods_names=cost, fig_name="./set1.2_fig/" + "{}_random_s05_run_{}_tcnt_{}_avg_accuracy.png".format(sd, time, target))
        plot_accuracies(ots, sorted_ot_series, final_losses, final_losses_var, methods_names=cost, fig_name="./set1.3_fig/" + "{}_sd2_{}_len{}_ot{}_seq{:.5f}_dire{:.5f}.png".format(sd, unorder_sd, time, ot_series, final_losses[0, -1], final_losses[1, -1]))

