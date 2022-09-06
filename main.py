from os import times_result
from random import random
# from typing import final
import numpy as np
import pandas as pd
# from torch import R
from da_ot import OTGroupLassoDAClassifier, OTBFBDAClassifier
from data import Xt_all, load_battery_data_split
from plot import plot_continuous_domain_adaptation, plot_accuracies

from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn import tree
import random
import ot
import pdb

def test_cdot_methods(methods, time_reg_vector, n_samples_source, n_samples_targets,
                      time_length, time_series, sort_method, if_sort, methods_names=None, cost="seq",
                      fig_name=None, plot_mapping=False, random_seed = 0):
    
    # Xs, ys, Xt, yt, Xt_all, yt_all = load_battery_data(n_samples_source, n_samples_targets, time_length, True)
    # Xs, ys, Xt, yt, Xt_all, yt_all = load_battery_data_random(n_samples_source, n_samples_targets, time_series, shuffle_or_not = True, random_seed = random_seed)
    Xs, ys, Xt, yt, Xt_all, yt_all, acc, Xt_true, yt_true, Xt_random, yt_random = load_battery_data_split(n_samples_source, n_samples_targets, time_series, shuffle_or_not = True, random_seed = random_seed, train_set = 20)

    if sort_method == 'w_dis':
        Xs, ys, Xt_w, yt_w, Xt_all, yt_all, acc, Xt_true_w, yt_true_w, _, _ = load_battery_data_split(n_samples_source, n_samples_targets, time_series, shuffle_or_not = True, random_seed = random_seed + 1000, train_set = 20)
        Xt_true_w.pop()
        yt_true_w.pop()

        Xt_w.pop()
        yt_w.pop()

        if if_sort == 1:
            w_dist = []
            for x in Xt_true_w:
                m = ot.dist(Xs, x, metric='euclidean')
                m /= m.max()
                n1 = Xs.shape[0]
                n2 = x.shape[0]
                a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2
                c = ot.sinkhorn2(a, b, m, 1)
                w_dist.append(c)
            Xt = [x for _, x in sorted(zip(w_dist, Xt_w), key=lambda x1: x1[0])]
            yt = [x for _, x in sorted(zip(w_dist, yt_w), key=lambda x1: x1[0])]

        if if_sort == 0:
            rand = np.arange(len(Xt)-1)
            np.random.seed(random_seed)
            np.random.shuffle(rand)
            Xt = [x for _, x in sorted(zip(rand, Xt_w), key=lambda x1: x1[0])]
            yt = [x for _, x in sorted(zip(rand, yt_w), key=lambda x1: x1[0])]

        Xt.append(Xt_true[-1])
        yt.append(yt_true[-1])
    
    if sort_method == 'soc':
        Xt = Xt_true
        yt = yt_true
    
    if sort_method == 'random':
        Xt = Xt_random
        yt = yt_random
    
    mapped_samples = []
    time_reg = []
    entropic_reg = []

    scores = np.zeros([len(methods), time_length])
    losses = np.zeros([len(methods), time_length])
    ots = np.arange(time_length)

    for m, da_clf in enumerate(methods):
        # print("Running ", methods_names[m])
        temp_samples = []
        for k in range(time_length):
            if k > 0:
                if cost == "seq":
                    da_clf.fit(temp_samples[-1], ys, Xt[k], treg=time_reg_vector[m], Gamma_old=da_clf.Gamma,
                               Xt_old=Xt[k - 1])

                    time_reg.append(da_clf.temp_reg(da_clf.Gamma))
                    # print("time_reg: ", time_reg)
                    entropic_reg.append(da_clf.entropic_reg(da_clf.Gamma))
                    # print("entropic reg: ", entropic_reg)
                    # print("--seq oting--")
                else:
                    # da_clf.fit(Xs, ys, Xt[k], treg=time_reg_vector[m], Gamma_old=da_clf.Gamma,
                    #            Xt_old=Xt[k - 1])
                    da_clf.fit(Xs, ys, Xt[k])
                    # print("--direct oting--")

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
  

    return scores, losses, ots, time_reg, entropic_reg, acc

if __name__ == '__main__':
    # sort_method = 'soc'
    # sort_method = 'clf'
    sort_method = 'w_dis'
    # sort_method = 'random'
    data = []
    for t in range(8):
        # # of target and intermediate domains
        time = t + 2
        print("============= # of intermediate domains: {} =============".format(time - 1))

        per_epoch_loss_order = []
        per_epoch_loss_direct = []
        per_epoch_loss_unorder = []
        sort_or_not = [1, 0]

        for if_sort in sort_or_not:
            print("----------- sort or not: {} ------------".format(if_sort))
            N_RUNS = 10

            # clf = KNeighborsClassifier(n_neighbors=1)
            # clf = ensemble.RandomForestRegressor(n_estimators=4)
            # clf = Ridge(alpha=5)
            clf = svm.SVR(gamma='scale')
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

            clf_acc = []

            for sd in range(100):
                print("#epoch {}".format(sd))
                np.random.seed(sd)
                # time = np.random.randint(1, 10)
                # ot_series = 5 * np.random.randint(2, 10, time)
                ot_series = 5 * (np.random.choice(9, time, replace=False) + 2)

                sorted_ot_series = ot_series

                
                # sorted_ot_series = sorted_ot_series[np.where(sorted_ot_series <= sorted_ot_series[-1])]
                sorted_ot_series = np.sort(sorted_ot_series)

                sample_num = len(sorted_ot_series) - 1
                unsorted_ot_series = sorted_ot_series
                if sample_num > 0 and time > 1:
                    unsorted_ot_series = 5 * (np.random.choice(9, sample_num, replace=False) + 2)
                    unsorted_ot_series = np.append(unsorted_ot_series, sorted_ot_series[-1])
                

                if if_sort == 1:
                    cost = ["seq", "direct"]
                else:
                    cost = ["seq"]
                    sorted_ot_series = unsorted_ot_series
                
                time = len(sorted_ot_series)
                target = 10

                final_scores = np.zeros([len(cost), time])
                final_losses = np.zeros([len(cost), time])

                final_scores_var = np.zeros([len(cost), time])
                final_losses_var = np.zeros([len(cost), time])

                for i, c in enumerate(cost):
                    # print('-----{} ot-----'.format(c))

                    run_scores = []
                    run_losses = []

                    for run in range(N_RUNS):
                        # print("RUN %d..." % run)
                        scores, losses, ots, time_reg, entropic_reg, acc = test_cdot_methods(
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
                            random_seed = (sd+1) * (run+1),
                            sort_method = sort_method,
                            if_sort=if_sort
                        )
                        run_scores.append(scores)
                        run_losses.append(losses)
                        
                        clf_acc.append(acc)

                    avg_scores = np.mean(np.array(run_scores), axis=0)
                    score_var = np.var(np.array(run_scores), axis=0)
                    
                    avg_losses = np.mean(np.array(run_losses), axis=0)
                    loss_var = np.var(np.array(run_losses), axis=0)

                    final_scores[i, :] = avg_scores
                    final_losses[i, :] = avg_losses

                    final_scores_var[i, :] = score_var
                    final_losses_var[i, :] = loss_var
                
                if if_sort == 1:
                    per_epoch_loss_order.append(final_losses[0, -1])
                    per_epoch_loss_direct.append(final_losses[1, -1])
                else:
                    per_epoch_loss_unorder.append(final_losses[0, -1])

                # print("ot series: ", ot_series)
                # print("sorted ot series: ", sorted_ot_series)
                # print("unsorted ot series: ", unsorted_ot_series)
                # print("average loss: ", final_losses[:, -1])
                # plot_accuracies(ots, sorted_ot_series, final_scores, final_scores_var, methods_names=cost, fig_name="./set1.4_fig/" + "{}_random_s05_run_{}_tcnt_{}_avg_accuracy.png".format(sd, time, target))
                # plot_accuracies(ots, sorted_ot_series, final_losses, final_losses_var, methods_names=cost, fig_name="./set1.4_fig/" + "{}_len{}_ot{}_seq{:.5f}_dire{:.5f}.png".format(sd, time, sorted_ot_series, final_losses[0, -1], final_losses[1, -1]))

            avg_clf_acc = np.mean(clf_acc)
            print(avg_clf_acc)

        mean_per_epoch_loss_order = np.mean(per_epoch_loss_order)
        mean_per_epoch_loss_direct = np.mean(per_epoch_loss_direct)
        mean_per_epoch_loss_unorder = np.mean(per_epoch_loss_unorder)
        var_per_epoch_loss_order = np.var(per_epoch_loss_order)
        var_per_epoch_loss_direct = np.var(per_epoch_loss_direct)
        var_per_epoch_loss_unorder = np.var(per_epoch_loss_unorder)
        print("mean of order ", mean_per_epoch_loss_order)
        print("mean of direct ", mean_per_epoch_loss_direct)
        print("mean of unorder ", mean_per_epoch_loss_unorder)
        print("var of order ", var_per_epoch_loss_order)
        print("var of direct ", var_per_epoch_loss_direct)
        print("var of unorder ", var_per_epoch_loss_unorder)

        data.append({'time': time-1, 'mean_direct': mean_per_epoch_loss_direct, 'mean_unorder': mean_per_epoch_loss_unorder, 'mean_order': mean_per_epoch_loss_order, \
                    'var_direct': var_per_epoch_loss_direct, 'var_unorder': var_per_epoch_loss_unorder, 'var_order': var_per_epoch_loss_order})

    dataframe = pd.DataFrame(data)
    dataframe.to_csv("soc_{}_clf_sametarget_ssh_{}.csv".format(sort_method, target), index=False, sep=',')    

        
