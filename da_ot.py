from telnetlib import GA
import numpy as np
import ot
from BFB import *
import pdb


def subgrad_norm(w):
    norm_w = np.linalg.norm(w)
    if norm_w > 0:
        g = w / norm_w
    else:
        g = np.ones_like(w)
    return g


class DAClassifier:

    def __init__(self, clf):
        self.clf = clf
        self.max_attempts = 3

    def adapt_source_to_target(self):
        return NotImplementedError

    def adapt_target_to_source(self):
        pass

    def fit(self, Xs=None, ys=None, Xt=None, treg=0, Gamma_old=None, Xt_old=None):
        self.Xs = Xs
        self.ys = ys
        self.Xt = Xt

        self.treg = treg
        self.Gamma_old = Gamma_old
        self.Xt_old = Xt_old

        attempt_count = 0
        while attempt_count < self.max_attempts:
            try:
                self.fit_domain_adapter()
                break
            except:
                attempt_count += 1
        
        Xs_mapped = self.adapt_source_to_target()
        self.clf.fit(Xs_mapped, self.ys)

    def grad_entropic_reg(self, Gamma):
        _grad_entropic_reg = np.log(Gamma)
        return _grad_entropic_reg
    
    def entropic_reg(self, Gamma):
        entropic_reg = np.sum(Gamma * np.log(Gamma) - Gamma)
        return entropic_reg
    
    ## --------------- ##
    ## Time regulizer ## 
    ## -------------- ##
    
    def temp_reg(self, Gamma):
        temp_reg = self.treg * np.linalg.norm((Gamma @ self.Xt - self.Gamma_old @ self.Xt_old), ord='fro') ** 2
        return temp_reg

    def grad_temp_reg(self, Gamma):
        _grad_temp_reg = 2 * self.treg * (Gamma @ self.Xt - self.Gamma_old @ self.Xt_old) @ self.Xt.T
        return _grad_temp_reg

    ## --------------- ##
    ## Lasso regulizer ## 
    ## --------------- ##

    def lasso_reg(self, Gamma):
        labels = np.unique(self.ys)
        res = 0
        for i in range(self.n_samples_target):
            for lab in labels:
                temp = Gamma[self.ys == lab, i]
                res += np.linalg.norm(temp, ord=2)
        return res

    def time_lasso_reg(self, Gamma):
        return self.lasso_reg(Gamma) + self.temp_reg(Gamma)

    def grad_time_lasso_reg(self, Gamma):
        return self.time_lasso_reg(Gamma) + self.grad_temp_reg(Gamma)

    def fit_domain_adapter(self):
        return NotImplementedError

    @property
    def n_samples_source(self):
        return self.Xs.shape[0]

    @property
    def n_samples_target(self):
        return self.Xt.shape[0]

    def predict_target(self):
        return self.clf.predict(self.Xt)

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, X, y):
        return self.clf.score(X, y)


class OTDAClassifier(DAClassifier):

    def adapt_source_to_target(self):
        return self.n_samples_source * self.Gamma @ self.Xt

    def adapt_target_to_source(self):
        return self.n_samples_target * np.transpose(self.Gamma) @ self.Xs

    def fit_domain_adapter(self):
        r = 1 / self.n_samples_source * np.ones(self.n_samples_source)
        c = 1 / self.n_samples_target * np.ones(self.n_samples_target)
        Cost = ot.dist(self.Xs, self.Xt, metric='sqeuclidean')
        if self.treg == 0:
            self.Gamma = ot.emd(r, c, Cost)
        else:
            self.Gamma = ot.optim.cg(r, c, Cost, self.treg, self.temp_reg, self.grad_temp_reg)


class OTSinkhornDAClassifier(DAClassifier):

    def __init__(self, clf, reg=0.01):
        self.reg = reg
        super(OTSinkhornDAClassifier, self).__init__(clf)

    def adapt_source_to_target(self):
        return self.n_samples_source * self.Gamma @ self.Xt

    def adapt_target_to_source(self):
        return self.n_samples_target * np.transpose(self.Gamma) @ self.Xs

    def fit_domain_adapter(self):
        r = 1 / self.n_samples_source * np.ones(self.n_samples_source)
        c = 1 / self.n_samples_target * np.ones(self.n_samples_target)
        Cost = ot.dist(self.Xs, self.Xt, metric='sqeuclidean')
        if self.treg == 0:
            self.Gamma, self.history = ot.sinkhorn(r, c, Cost, self.reg, log=True)
        else:
            self.Gamma, self.history = ot.optim.gcg(r, c, Cost, self.reg, self.treg, self.temp_reg, self.grad_temp_reg, log=True)


class OTKernelMappingDAClassifier(DAClassifier):

    def __init__(self, clf, mu=1, eta=0.001, kernel="gaussian", sigma=1, bias=False):
        self.mapping_transport = ot.da.MappingTransport(kernel=kernel,
                                                        mu=mu,
                                                        eta=eta,
                                                        sigma=sigma)
        super(OTKernelMappingDAClassifier, self).__init__(clf)

    def adapt_source_to_target(self):
        return self.mapping_transport.transform(Xs=self.Xs)

    def fit_domain_adapter(self):
        self.mapping_transport.fit(Xs=self.Xs, Xt=self.Xt)


class OTLinearMappingDAClassifier(OTKernelMappingDAClassifier):

    def __init__(self, clf, mu=1, eta=0.001):
        super(OTLinearMappingDAClassifier, self).__init__(clf, mu=mu, eta=eta, kernel="linear", bias=True)


class OTGroupLassoDAClassifier(DAClassifier):

    def __init__(self, clf, reg=0.01, eta=0.01):
        self.reg = reg
        self.eta = eta
        super(OTGroupLassoDAClassifier, self).__init__(clf)

    def adapt_source_to_target(self):
        return self.n_samples_source * self.Gamma @ self.Xt

    def adapt_target_to_source(self):
        return self.n_samples_target * np.transpose(self.Gamma) @ self.Xs

    def fit_domain_adapter(self):
        r = 1 / self.n_samples_source * np.ones(self.n_samples_source)
        c = 1 / self.n_samples_target * np.ones(self.n_samples_target)
        Cost = ot.dist(self.Xs, self.Xt, metric='sqeuclidean')
        if self.treg == 0:
            self.Gamma = ot.da.sinkhorn_l1l2_gl(r, self.ys, c, Cost, reg=self.reg, eta=self.eta)
        else:
            self.Gamma = ot.optim.gcg(r, c, Cost, self.reg, self.eta, self.time_lasso_reg, self.grad_time_lasso_reg)


class OTBFBDAClassifier(DAClassifier):

    def __init__(self, clf, reg, regnorm, it, epochs, lr, verbose=True):
        self.reg = reg
        self.regnorm = regnorm
        self.it = it
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        super(OTBFBDAClassifier, self).__init__(clf)

    def adapt_source_to_target(self):
        return self.n_samples_source * self.Gamma @ self.Xt

    def adapt_target_to_source(self):
        return self.n_samples_target * np.transpose(self.Gamma) @ self.Xs

    def fit_domain_adapter(self):
        r = 1 / self.n_samples_source * np.ones(self.n_samples_source)
        c = 1 / self.n_samples_target * np.ones(self.n_samples_target)
        Cost = ot.dist(self.Xs, self.Xt, metric='sqeuclidean')

        if self.treg == 0:
            self.Gamma, self.history = ot.sinkhorn(r, c, Cost, self.reg, log=self.verbose)
        else:
            #self.Gamma = optimize_BFB_timereg(r, c, Cost, self.reg, self.treg, self.regnorm, self.Gamma_old, self.Xt,
            #                                  self.Xt_old, self.it, self.epochs, self.lr, seed=42, verbose=True)
            self.Gamma, self.history = gcg_proximal(r, c, Cost, self.lr, self.reg, self.grad_entropic_reg, self.temp_reg, self.grad_temp_reg, verbose=False, log=self.verbose)
