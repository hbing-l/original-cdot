from os import times_result
from random import random
# from typing import final
import numpy as np
import pandas as pd
# from torch import R
from da_ot import OTGroupLassoDAClassifier, OTBFBDAClassifier
from data import Xt_all, Xt_all_domain, load_battery_data_split, load_seq_two_moon_data_test_1
from plot import plot_continuous_domain_adaptation, plot_accuracies

from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

import random
import ot
import pdb

def test_cdot_methods(methods, time_reg_vector, n_samples_source, n_samples_targets,
                      time_length, sort_method, methods_names=None, cost="seq",
                      fig_name=None, plot_mapping=False, random_seed = 0):
    
    # Xs, ys, Xt, yt, Xt_all, yt_all = load_battery_data(n_samples_source, n_samples_targets, time_length, True)
    # Xs, ys, Xt, yt, Xt_all, yt_all = load_battery_data_random(n_samples_source, n_samples_targets, time_series, shuffle_or_not = True, random_seed = random_seed)
    # Xs, ys, Xt, yt, Xt_all, yt_all, acc, Xt_true, yt_true, Xt_random, yt_random, Xt_all_domain, yt_all_domain, Xt_all_domain_mix, yt_all_domain_mix = load_battery_data_split(n_samples_source, n_samples_targets, time_series, shuffle_or_not = True, random_seed = random_seed, train_set = 20)
    
    Xs, ys, Xt, yt, angles, Xt_all_domain, yt_all_domain, Xt_random, yt_random = load_seq_two_moon_data_test_1(n_samples_source, n_samples_targets, time_length, noise=0.1)

    _, _, Xtest, ytest, _, _, _, _, _ = load_seq_two_moon_data_test_1(0, 500, time_length, noise=0.1)


    if sort_method == 'w_dis':

        m = ot.dist(Xs, Xt[-1], metric='euclidean')
        m /= m.max()
        n1 = Xs.shape[0]
        n2 = Xt[-1].shape[0]
        a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2
        c_t = ot.sinkhorn2(a, b, m, 1)

        w_dist = []
        for x in Xt_all_domain:
            m = ot.dist(Xs, x, metric='euclidean')
            m /= m.max()
            n1 = Xs.shape[0]
            n2 = x.shape[0]
            a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2
            c = ot.sinkhorn2(a, b, m, 1)
            w_dist.append(c)
        
        t = time_length - 1
        w_dist_min = []
        Xt_min = []
        yt_min = []
        for i in range(len(w_dist)):
            if w_dist[i] <= c_t:
                w_dist_min.append(w_dist[i])
                Xt_min.append(Xt_all_domain[i])
                yt_min.append(yt_all_domain[i])

        
        rand = np.arange(len(w_dist_min))
        np.random.seed(random_seed)
        np.random.shuffle(rand)

        Xt1 = [x for _, x in sorted(zip(rand, Xt_min), key=lambda x1: x1[0])]
        yt1 = [x for _, x in sorted(zip(rand, yt_min), key=lambda x1: x1[0])]
        w1 = [x for _, x in sorted(zip(rand, w_dist_min), key=lambda x1: x1[0])]
        
        if t <= len(Xt1):
            Xt1 = Xt1[:t]
            yt1 = yt1[:t]
            w1 = w1[:t]
            Xt_ = [x for _, x in sorted(zip(w1, Xt1), key=lambda x1: x1[0])]
            yt_ = [x for _, x in sorted(zip(w1, yt1), key=lambda x1: x1[0])]

        elif len(Xt1) == 0:
            
            x_sample = []
            y_sample = []
            for i in range(t):
                x_sample_per = np.array(Xt_all_domain[i])
                y_sample_per = np.array(yt_all_domain[i])
                
                x_sample.append(x_sample_per)
                y_sample.append(y_sample_per)

            Xt_ = x_sample
            yt_ = y_sample

        else:
            x_sample = []
            y_sample = []
            w_sample = []
            for i in range(t - len(Xt1)):
                np.random.seed(1 * i)
                rand = np.random.choice(len(Xt1), 1)
                x_sample_per = np.array(Xt1[rand[0]])
                y_sample_per = np.array(yt1[rand[0]])
                w_sample_per = np.array(w1[rand[0]])
                
                x_sample.append(x_sample_per)
                y_sample.append(y_sample_per)
                w_sample.append(w_sample_per)

            Xt1 = Xt1 + x_sample
            yt1 = yt1 + y_sample
            w1 = w1 + w_sample
            
            Xt_ = [x for _, x in sorted(zip(w1, Xt1), key=lambda x1: x1[0])]
            yt_ = [x for _, x in sorted(zip(w1, yt1), key=lambda x1: x1[0])]

        Xt_.append(Xt[-1])
        yt_.append(yt[-1])

        Xt = Xt_
        yt = yt_

    
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
            
            result = da_clf.predict(Xtest[k])
            scores[m, k] = r2_score(ytest[k], result)
            losses[m, k] = da_clf.score(Xtest[k], ytest[k])

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
  

    return scores, losses, ots, time_reg, entropic_reg

if __name__ == '__main__':
    # sort_method = 'soc'
    # sort_method = 'clf'
    # sort_method = 'w_dis'
    sort_method = 'random'
    # sort_method = 'mix'
    data = []
    print("sort method: ".format(sort_method))
    for t in range(8):
        # # of target and intermediate domains
        angle = (t + 1) * 18
        print("============= target angle: {} =============".format(angle))

        per_epoch_loss_order = []
        per_epoch_loss_direct = []
        per_epoch_loss_unorder = []
        sort_or_not = [1, 0]

        N_RUNS = 1

        clf = KNeighborsClassifier(n_neighbors=1)
        # clf = ensemble.RandomForestRegressor(n_estimators=4)
        # clf = Ridge(alpha=5)
        # clf = svm.SVR(gamma='scale')
        # clf = tree.DecisionTreeRegressor()


        ot_group_lasso = OTGroupLassoDAClassifier(clf, reg=0.5, eta=10)
        ot_group_lasso_treg = OTGroupLassoDAClassifier(clf, reg=0.5, eta=10)
        ot_BFB = OTBFBDAClassifier(clf, reg=0.1, regnorm=None, it=50, epochs=1000, lr=10, verbose=True)


        methods = [ot_BFB]
        methods_names =  ['ot_BFB']
        time_reg_vector = [50]

        cost = ["seq", "direct"]

        # clf_acc = []

        for sd in range(10):
            print("#epoch {}".format(sd))
            np.random.seed(sd)
            
            time = t + 1
            target = 100

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
                    scores, losses, ots, time_reg, entropic_reg = test_cdot_methods(
                        methods=methods,
                        methods_names=methods_names,
                        time_reg_vector=time_reg_vector,
                        fig_name="seq_run_" + str(run),
                        time_length=time,
                        n_samples_source=500,
                        n_samples_targets=target,
                        plot_mapping=False,
                        cost=c,
                        random_seed = (sd+1) * (run+1),
                        sort_method = sort_method,
                    )
                    run_scores.append(scores)
                    run_losses.append(losses)
                    
                    # clf_acc.append(acc)

                avg_scores = np.mean(np.array(run_scores), axis=0)
                score_var = np.var(np.array(run_scores), axis=0)
                
                avg_losses = np.mean(np.array(run_losses), axis=0)
                loss_var = np.var(np.array(run_losses), axis=0)

                final_scores[i, :] = avg_scores
                final_losses[i, :] = avg_losses

                final_scores_var[i, :] = score_var
                final_losses_var[i, :] = loss_var
            
            per_epoch_loss_order.append(final_losses[0, -1])
            per_epoch_loss_direct.append(final_losses[1, -1]) 

        mean_per_epoch_loss_order = np.mean(per_epoch_loss_order)
        mean_per_epoch_loss_direct = np.mean(per_epoch_loss_direct)
        var_per_epoch_loss_order = np.var(per_epoch_loss_order)
        var_per_epoch_loss_direct = np.var(per_epoch_loss_direct)
        print("mean of order ", mean_per_epoch_loss_order)
        print("mean of direct ", mean_per_epoch_loss_direct)
        print("var of order ", var_per_epoch_loss_order)
        print("var of direct ", var_per_epoch_loss_direct)

        data.append({'time': time-1, 'mean_direct': mean_per_epoch_loss_direct, 'mean_order': mean_per_epoch_loss_order, \
                    'var_direct': var_per_epoch_loss_direct, 'var_order': var_per_epoch_loss_order})

    dataframe = pd.DataFrame(data)
    dataframe.to_csv("half_{}_sametarget_fix_t_domain_ssh_{}_e10_n1.csv".format(sort_method, target), index=False, sep=',')    

        
