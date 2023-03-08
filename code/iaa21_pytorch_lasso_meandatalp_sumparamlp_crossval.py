

# %%


def get_predictions(traindatain, opt_param, seed=None):
    import math
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.distributions import Normal, MultivariateNormal, Laplace, TransformedDistribution
    from torch.optim import Adam
    from torch.distributions.transforms import ComposeTransform, AffineTransform, SigmoidTransform
    import time

    def inverse_soft_plus(x):
        import math
        return math.log(math.exp(x) - 1)

    def inverse_soft_plus_torch(x):
        return torch.log(torch.exp(x) - 1)

    class Lin(nn.Module):
        def __init__(self, n_emotions, n_features, opt_dict=None):  # , Laplace_scale=1, L1_constant=0
            super().__init__()

            self.data = None

            ### NB no prior on sigma and b (and k) ###

            ### init around zero ###
            self.b = nn.Parameter(0.01 * torch.randn(n_emotions))
            ### init around 1 ###
            self._sigma_emotions = nn.Parameter(inverse_soft_plus(1.0) * torch.ones(n_emotions) + 0.01 * torch.randn(n_emotions))

            self.logit_k_param = opt_dict.get('logit_k', 1.0)
            if self.logit_k_param == 'fit':
                ### init around 1 ###
                self._logit_k = nn.Parameter(inverse_soft_plus(1.0) * torch.ones(1) + 0.01 * torch.randn(1))
            else:
                self._logit_k = self.logit_k_param * torch.ones(1)

            ### init around 0 ###
            self.A = nn.Parameter(0.01 * torch.randn(n_emotions, n_features))

            if opt_dict['kind'] != 'noReg':
                self.regularize = True
                if opt_dict['kind'] == 'LASSO':
                    self.regularizer = opt_dict['kind']
                    self.l1_scale = opt_dict['l1_scale']
                    self.laplace_score = Laplace(0, self.l1_scale)
            else:
                self.regularizer = 'NONE'
                self.regularize = False

        @property
        def sigma_emotions(self):
            return F.softplus(self._sigma_emotions)

        @property
        def logit_k(self):
            if self.logit_k_param == 'fit':
                logit_k_ = F.softplus(self._logit_k, beta=1)
            else:
                logit_k_ = self._logit_k
            return logit_k_

        # def logit_affine(self):  # [0,1]->(-inf,inf) :: affine compress by b -> logit -> affine stretch by 1/k
        #     affine_intercept = self.affine_b
        #     affine_slope = 1.0 - 2.0 * self.affine_b
        #     return ComposeTransform([AffineTransform(loc=affine_intercept, scale=affine_slope), SigmoidTransform().inv, AffineTransform(loc=torch.Tensor([0.0]), scale=1.0 / self.logit_k)])

        @property
        def invaffine_logistic(self):
            return ComposeTransform([AffineTransform(loc=torch.Tensor([0.0]), scale=self.logit_k), SigmoidTransform()])

        """
        use affine stretch for logistic(mu_logit), lp = N( yobs | affineStretch( logistic( mu_logit, k ), b ) )
        """

        def score_single_dataset(self, x, y, jy):
            ### xsize :: torch.Size([96, 375, 19])   --  [(24 pots * 4 outcomes), samples per condition, iaf]
            ### mu: torch.Size([96, 375, 20]) -- predicted vector of intensities corresponding to each sample
            mu_logit = (x[:, :, None, :] * self.A[None, None, :, :]).sum(dim=3) + self.b[None, None, :]
            ### this assumes that the first dimension of x corresponseds to stim id [1,2,3,4...n]
            ### i.e. maps the jy id to an x row

            ### transform predicted values into [0,1] space of empirical data
            ### if self.affine_b == 0: 0 < mu_logistic < 1
            mu_logistic = self.invaffine_logistic(mu_logit)

            dist = Normal(mu_logistic[jy], self.sigma_emotions[None, None, :])

            ### calculate the probability density of observing \vec{e_i} in the mixture
            ### i.e. score the 20d vector y[i,None,0:19] in the Gaussian over mu_logistic[jy[i],j,0:19]
            ### where j is an index of webppl samples and jy[i] gives the stimulus ID of \vec{e_i}
            score = dist.log_prob(y[:, None, :]).sum(dim=2)
            ### multiply the probability densities of each dimension of the 20d empirical emotion vector to give a probability for the empirical vector in each webppl sample corresponding to that stimulus (e.g. score might be 4432 empirical observations x 375 webppl samples)

            #### score = score.logsumexp(dim=1) - math.log(score.size(1))

            return score

        def score(self):
            loglik_data_list = [self.score_single_dataset(data['X'], data['Y'], data['Jy']) for data in self.data]

            ### NOTE think about weights of losses (adjusting for number of empirical observations in each dataset, etc.)

            ### stack (observations x webppl samples) lp across datasets (now n_total_empirical_observations x n_webpplsamples)
            ### where each element is the lp of observing a given 20d empirical vector in a given 20d gaussian
            loglik_data_temp = torch.cat(loglik_data_list, dim=0)

            ### score.logsumexp(dim=1) <- for each empirical vector, sum the probabilities (not the log probabilities) of observing that vector in all the webppl normals
            ### divide the total by the number of webppl samples to normalize
            ### take the mean of all the empirical obserations to yield the probability of observing the empirical data in the mixture
            ### taking the mean rather than the sum of the logprobs allows for comparision between datasets with different numbers of empirical observations

            ### loglik_data_temp <n_total_empirical_observations, n_wppl_samples>, loglik_data_temp.logsumexp(dim=1) <n_total_empirical_observations>, loglik_data <1>
            loglik_data = (loglik_data_temp.logsumexp(dim=1) - math.log(loglik_data_temp.size(1))).mean()

            lp_prior = torch.Tensor([0.0])
            lp_prior += self.laplace_score.log_prob(self.A).sum()

            score = loglik_data + lp_prior

            return score

        def get_param(self):
            dict_out = dict()
            for key_, var_ in {
                'A': self.A.detach(),
                'b': self.b.detach(),
                'sigma_emotions': self.sigma_emotions.detach(),
                'logit_k': self.logit_k.detach(),
            }.items():  # .detach().item():
                dict_out[key_] = torch.empty_like(var_, requires_grad=False).copy_(var_).detach()
            return dict_out

    """
    right now, Y data stay raw in [0,1] space.
    X are transformed w/ standard scaler, prospect scale, etc. in data_transform
    mu_logit are transformed into mu_logistic with no affine
    a gaussian is placed on each mu_logistic, so while mu_logistic never reaches 0 or 1, the gaussian in which y is scored is unbounded.
    """

    class ModelTrainer():
        def __init__(self, datain):

            firstkey = list(datain.keys())[0]

            ### initialize on columns of Y and X, which should be consistant
            nemotions = datain[firstkey]['Ylong'].shape[1]
            nfeatures = datain[firstkey]['Xlong'].shape[2]
            self.train_data = list()
            for data in datain.values():
                self.train_data.append(dict(X=torch.Tensor(data['Xlong']), Y=torch.Tensor(data['Ylong']), Jy=torch.Tensor(data['Jy']).long()))

            self.model = Lin(nemotions, nfeatures, opt_param)
            self.model.data = self.train_data
            self.optimizer = Adam(self.model.parameters(), lr=0.01)
            self.istep = -1

            self.fit_log = list()
            self.score_ = None

        def step_optimizer(self):
            self.istep += 1
            self.optimizer.zero_grad()
            score = self.model.score()
            (-score).backward()
            self.optimizer.step()
            self.score_ = score.detach().mean().item()

        def apply(self, model_param, X_in):
            return apply_fit(model_param, X_in)

        def logger(self):
            self.fit_log.append(dict(param=self.model.get_param(), score=self.score_, iiter=self.istep))

        def get_current_param(self):
            return self.model.get_param()

        def get_fit_logger(self):
            from copy import deepcopy
            return deepcopy(self.fit_log)

    ### initialize on columns of Y and X, which should be consistant
    # nemotions = data_list[0]['Y'].shape[1]
    # nfeatures = data_list[0]['X'].shape[2]
    # model = Lin(nemotions, nfeatures, opt_param)

    seed_inherited = True
    if seed is None:
        seed = int(str(int(time.time() * 10**6))[-9:])
        seed_inherited = False
    torch.manual_seed(seed)

    return ModelTrainer(traindatain)


def apply_fit(fit_param, X_in):
    import torch
    from torch.distributions.transforms import ComposeTransform, AffineTransform, SigmoidTransform

    def invaffine_logistic(logit_k):
        return ComposeTransform([AffineTransform(loc=torch.Tensor([0.0]), scale=logit_k), SigmoidTransform()])

    invaffine_logistic_transform = invaffine_logistic(fit_param['logit_k'])

    x = torch.Tensor(X_in)

    A_ = fit_param['A'][None, None, :, :]

    x_ = x[:, :, None, :]
    b_ = fit_param['b'][None, None, :]

    mu_logit = (x_ * A_).sum(dim=3) + b_

    mu_logistic = invaffine_logistic_transform(mu_logit)

    return mu_logistic.detach().numpy()


def calc_cv_test_temp_dist(fit_param, torchdata_cv_test, withdata=False):
    import numpy as np
    from iaa_utils import concordance_corr_

    # torchdata_cv_train
    # torchdata_cv_test['generic'].keys()
    """
    fit_param = modelfit_log[-1]['param']
    """
    # torchdata_cv_test.keys()

    outcomes = ['CC', 'CD', 'DC', 'DD']

    emoev = dict()
    for stimid, data in torchdata_cv_test.items():
        ### prediction EV of training data for each pot-outcome
        Yhat = apply_fit(fit_param, data['X'])  # <32 pot-outcome, 375 sample, 20 emotion>
        Yhat_ev = np.mean(Yhat, axis=1)  # 32 <pot-outcome, 20 emotion>

        Yhat_ev_outcome_list = list()
        for outcome in ['CC', 'CD', 'DC', 'DD']:
            outcomeidx = data['Jxmap'].loc[data['Jxmap']['outcome'] == outcome, 'J'].to_numpy().astype(int)
            Yhat_ev_outcome_list.append(np.mean(np.mean(Yhat[outcomeidx, :], axis=1), axis=0))
        Yhat_ev_outcome = np.vstack(Yhat_ev_outcome_list)

        ### empirical EV of training data for each pot-outcome
        Y_ev = np.full([len(np.unique(data['Jy'])), data['Y'].shape[1]], np.nan, dtype=float)  # <96 pot-outcome, 20 emotion>
        for jy in range(len(np.unique(data['Jy']))):
            Y_ev[jy, :] = np.mean(data['Y'][data['Jy'] == jy, :], axis=0)  # sum over responses to a given pot-outcome
        assert not np.any(np.isnan(Y_ev))

        Y_ev_outcome_list = list()
        for outcome in outcomes:
            outcomeidx = data['Jymap'].loc[data['Jymap']['outcome'] == outcome, 'J'].to_numpy().astype(int)
            Y_ev_outcome_list.append(np.mean(data['Y'][outcomeidx, :], axis=0))
        Y_ev_outcome = np.vstack(Y_ev_outcome_list)

        emoev[stimid] = dict(
            Yhat_ev_potoutcome=Yhat_ev,
            Y_ev_potoutcome=Y_ev,
            Yhat_ev_outcome=Yhat_ev_outcome,
            Y_ev_outcome=Y_ev_outcome,
        )

    res_player = dict()
    alldeltas_model, alldeltas_empir = list(), list()
    allintenities_model, allintenities_empir = list(), list()
    for stimid, data in emoev.items():
        if stimid != 'generic':
            deltas_model_potoutcome = data['Yhat_ev_potoutcome'] - emoev['generic']['Yhat_ev_potoutcome']
            deltas_empir_potoutcome = data['Y_ev_potoutcome'] - emoev['generic']['Y_ev_potoutcome']

            deltas_model_outcome = data['Yhat_ev_outcome'] - emoev['generic']['Yhat_ev_outcome']
            deltas_empir_outcome = data['Y_ev_outcome'] - emoev['generic']['Y_ev_outcome']

            alldeltas_model.extend(deltas_model_outcome.flatten().tolist())
            alldeltas_empir.extend(deltas_empir_outcome.flatten().tolist())

            allintenities_model.extend(data['Yhat_ev_potoutcome'].flatten().tolist())
            allintenities_empir.extend(data['Y_ev_potoutcome'].flatten().tolist())

            res_player[stimid] = dict(
                deltas=dict(
                    bypotoutcome=concordance_corr_(deltas_model_potoutcome.flatten(), deltas_empir_potoutcome.flatten()),
                    byoutcome=concordance_corr_(deltas_model_outcome.flatten(), deltas_empir_outcome.flatten())
                )
            )
    res_overall = dict(
        specific_test_deltas_byoutcome=concordance_corr_(alldeltas_model, alldeltas_empir),
        specific_test_absinten_bypotoutcome=concordance_corr_(allintenities_model, allintenities_empir),
    )

    fitres_data = None
    if withdata:
        n_pots_test = 8
        n_emotions = 20
        pots = torchdata_cv_test['generic']['Jxmap']['pot'].unique()
        assert len(pots) == n_pots_test
        absint_outcomepot_empir = np.full([len(res_player.keys()), len(outcomes), n_pots_test, n_emotions], np.nan, dtype=float)
        absint_outcomepot_model = absint_outcomepot_empir.copy()
        deltas_outcomepot_empir = np.full([len(res_player.keys()), len(outcomes), n_emotions], np.nan, dtype=float)
        deltas_outcomepot_model = deltas_outcomepot_empir.copy()
        for i_stimid, stimid in enumerate(res_player.keys()):
            data = torchdata_cv_test[stimid]
            for i_outcome, outcome in enumerate(outcomes):
                outcomeidx = data['Jxmap']['outcome'] == outcome
                for i_pot, pot in enumerate(pots):
                    potidx = data['Jxmap']['pot'] == pot
                    jx = data['Jxmap'].loc[(outcomeidx) & (potidx), 'J'].to_numpy().astype(int)
                    absint_outcomepot_empir[i_stimid, i_outcome, i_pot, :] = emoev[stimid]['Y_ev_potoutcome'][jx, :]
                    absint_outcomepot_model[i_stimid, i_outcome, i_pot, :] = emoev[stimid]['Yhat_ev_potoutcome'][jx, :]

            deltas_outcomepot_empir[i_stimid, :, :] = emoev[stimid]['Y_ev_outcome'] - emoev['generic']['Y_ev_outcome']
            deltas_outcomepot_model[i_stimid, :, :] = emoev[stimid]['Yhat_ev_outcome'] - emoev['generic']['Yhat_ev_outcome']

        assert not np.any(np.isnan(absint_outcomepot_empir))
        assert not np.any(np.isnan(absint_outcomepot_model))
        assert not np.any(np.isnan(deltas_outcomepot_empir))
        assert not np.any(np.isnan(deltas_outcomepot_model))

        fitres_data = dict(
            test_outcomepot=dict(
                empir=absint_outcomepot_empir,  # <20 specific_player, 4 outcome, 8 pot, 20 emotion>
                model=absint_outcomepot_model,  # <20 specific_player, 4 outcome, 8 pot, 20 emotion>
            ),
            deltas=dict(
                empir=deltas_outcomepot_empir,  # <20 specific_player, 4 outcome, 20 emotion>
                model=deltas_outcomepot_model,  # <20 specific_player, 4 outcome, 20 emotion>
            )
        )

    return res_overall, fitres_data  # , res_player


def calc_fit_old(apply_fit_fn, fit_param, stan_data, data_dims=None):
    import numpy as np
    from iaa_utils import concordance_corr_

    fitres = dict(
        train=dict(
            byoutcomepot=dict(
                concordance=None,
            )
        ),
        test=dict(
            byoutcomepot=dict(
                concordance=None,
            )
        ),
        deltas=dict(
            byoutcome=dict(
                concordance=None,
            )
        ),
    )

    #########################
    ### prediction of training data
    #########################

    X = np.full([len(np.unique(stan_data['Jx'])), stan_data['n_samples'], stan_data['n_features']], np.nan, dtype=float)  # <96 pot-outcome, 20 emotion>
    for jj in range(len(np.unique(stan_data['Jx']))):
        X[jj, :, :] = stan_data['X'][stan_data['Jx'] == jj, :]
    assert not np.any(np.isnan(X))

    ### prediction EV of training data for each pot-outcome
    Yhat_train = apply_fit_fn(fit_param, X)  # <96 pot-outcome, 375 sample, 20 emotion>
    Yhat_train_EV = np.mean(Yhat_train, axis=1)  # 96 <pot-outcome, 20 emotion>

    ### empirical EV of training data for each pot-outcome
    Y_train_EV = np.full([len(np.unique(stan_data['Jy'])), len(data_dims['Y']['emotion'])], np.nan, dtype=float)  # <96 pot-outcome, 20 emotion>
    for jj in range(len(np.unique(stan_data['Jy']))):
        Y_train_EV[jj, :] = np.mean(stan_data['Y'][stan_data['Jy'] == jj, :], axis=0)  # sum over responses to a given pot-outcome
    assert not np.any(np.isnan(Y_train_EV))

    ### fit to absolute intensities of training data for each pot-outcome
    from scipy.stats import pearsonr
    fitres['train']['byoutcomepot']['mse'] = np.mean(np.square(np.subtract(Y_train_EV.flatten(), Yhat_train_EV.flatten())))
    fitres['train']['byoutcomepot']['pearson'] = pearsonr(Y_train_EV.flatten(), Yhat_train_EV.flatten())[0]
    fitres['train']['byoutcomepot']['concordance'] = concordance_corr_(Y_train_EV.flatten(), Yhat_train_EV.flatten())

    #########################
    ### prediction of test data
    #########################

    ### get model predictions for specific players, for the 8 pots
    ### prediction of absolute emo intensities of specific players (by outcome-pot)

    X_test = stan_data['X_test']  # <20 specific_player, 4 outcome, 8 pot, 375 sample, 19 feature>

    Yhat_test_EV = np.full(stan_data['Yexpectation_test_outcome_pot'].shape, np.nan, dtype=float)  # <20 specific_player, 4 outcome, 8 pot, 1 sample, 20 emotion>
    for i_player, player in enumerate(data_dims['Yexpectation_test_outcome_pot']['player']):
        for i_outcome, outcome in enumerate(data_dims['Yexpectation_test_outcome_pot']['outcome']):
            for i_pot, pot in enumerate(data_dims['Yexpectation_test_outcome_pot']['pot']):
                Yhat_test_EV[i_player, i_outcome, i_pot, 0, :] = np.mean(np.squeeze(apply_fit_fn(fit_param, np.expand_dims(X_test[i_player, i_outcome, i_pot, :, :], axis=0))), axis=0)
    assert not np.any(np.isnan(Yhat_test_EV))

    fitres['test']['byoutcomepot']['mse'] = np.mean(np.square(np.subtract(stan_data['Yexpectation_test_outcome_pot'].flatten(), Yhat_test_EV.flatten())))
    fitres['test']['byoutcomepot']['pearson'] = pearsonr(stan_data['Yexpectation_test_outcome_pot'].flatten(), Yhat_test_EV.flatten())[0]
    fitres['test']['byoutcomepot']['concordance'] = concordance_corr_(stan_data['Yexpectation_test_outcome_pot'].flatten(), Yhat_test_EV.flatten())

    #########################
    ### prediction of deltas (by outcome)
    #########################

    ### get model predictions for generic players, for the 8 pots ###
    X_trainTest = stan_data['X_trainTest']  # <1 generic_player, 4 outcome, 8 pot, 375 sample, 19 feature>

    Yhat_train_EV_reducedpots = np.full(stan_data['Yexpectation_trainTest_outcome_pot'].shape, np.nan, dtype=float)  # <1 generic_player, 4 outcome, 8 pot, 1 sample, 20 emotion>
    for i_player, player in enumerate(data_dims['Yexpectation_trainTest_outcome_pot']['player']):
        for i_outcome, outcome in enumerate(data_dims['Yexpectation_trainTest_outcome_pot']['outcome']):
            for i_pot, pot in enumerate(data_dims['Yexpectation_trainTest_outcome_pot']['pot']):
                Yhat_train_EV_reducedpots[i_player, i_outcome, i_pot, :] = np.mean(
                    np.squeeze(
                        apply_fit_fn(
                            fit_param,
                            np.expand_dims(X_trainTest[i_player, i_outcome, i_pot, :, :], axis=0)  # add player dimension --> <1 player, 375 sample, 19 iaf>
                        )  # <1 player, 375 sample, 20 emotion>
                    ),  # remove player dimension --> <375 sample, 20 emotion>
                    axis=0)  # sum over samples --> <1 sample, 20 emotion>
    assert not np.any(np.isnan(Yhat_train_EV_reducedpots))

    ### Empirical deltas (by outcome) ###

    ### empirical emotion EV
    Y_train_EVoutcome = np.mean(stan_data['Yexpectation_trainTest_outcome_pot'], axis=2, keepdims=True)  # sum over pots --> <1 generic_player, 4 outcome, 1 pot, 1 sample, 20 emotion>
    Y_test_EVoutcome = np.mean(stan_data['Yexpectation_test_outcome_pot'], axis=2, keepdims=True)  # sum over pots --> <20 specific_player, 4 outcome, 1 pot, 1 sample, 20 emotion>

    ### model emotion EV
    Yhat_train_EVoutcome = np.mean(Yhat_train_EV_reducedpots, axis=2, keepdims=True)  # sum over pots --> <1 generic_player, 4 outcome, 1 pot, 1 sample, 20 emotion>
    Yhat_test_EVoutcome = np.mean(Yhat_test_EV, axis=2, keepdims=True)  # sum over pots --> <20 specific_player, 4 outcome, 1 pot, 1 sample, 20 emotion>

    players_test = data_dims['Yexpectation_test_outcome_pot']['player']
    outcomes = data_dims['Yexpectation_test_outcome_pot']['outcome']
    emotions = data_dims['Yexpectation_test_outcome_pot']['emotion']
    deltas_empir = np.full([len(players_test), len(outcomes), len(emotions)], np.nan, dtype=float)
    deltas_model = np.full_like(deltas_empir, np.nan, dtype=float)
    for i_player, player in enumerate(players_test):
        deltas_empir[i_player, :, :] = Y_test_EVoutcome[i_player, :, 0, 0, :] - Y_train_EVoutcome[0, :, 0, 0, :]
        deltas_model[i_player, :, :] = Yhat_test_EVoutcome[i_player, :, 0, 0, :] - Yhat_train_EVoutcome[0, :, 0, 0, :]
    assert not np.any(np.isnan(deltas_empir))
    assert not np.any(np.isnan(deltas_model))

    fitres['deltas']['byoutcome']['mse'] = np.mean(np.square(np.subtract(deltas_empir.flatten(), deltas_model.flatten())))
    fitres['deltas']['byoutcome']['concordance'] = concordance_corr_(deltas_empir.flatten(), deltas_model.flatten())

    fitres_data = dict(
        test_outcomepot=dict(
            empir=np.squeeze(stan_data['Yexpectation_test_outcome_pot']),  # remove samples dim --> <20 specific_player, 4 outcome, 8 pot, 20 emotion>
            model=np.squeeze(Yhat_test_EV)),  # remove samples dim --> <20 specific_player, 4 outcome, 8 pot, 20 emotion>
        deltas=dict(
            empir=deltas_empir,  # <20 specific_player, 4 outcome, 20 emotion>
            model=deltas_model),  # <20 specific_player, 4 outcome, 20 emotion>
    )

    return fitres, fitres_data


def calc_fit(fit_param, datatest, withdata=False):
    import numpy as np
    from iaa_utils import concordance_corr_

    # torchdata_cv_train
    # torchdata_cv_test['generic'].keys()
    """
    fit_param = modelfit_log[-1]['param']
    """
    # torchdata_cv_test.keys()

    """
    stimid = 'generic'
    datatest[stimid].keys()
    ['Xshort', 'Xshortdims', 'Yshort', 'Yshortdims', 'Xlong', 'Ylong', 'Jy', 'Jxmap', 'Jymap']
    """

    emoev = dict()
    for stimid, data in datatest.items():

        n_emotions = data['Ylong'].shape[1]
        yhat_ev = np.full([len(data['Xshortdims']['outcome']), len(data['Xshortdims']['pot']), n_emotions], np.nan, dtype=float)
        y_ev = yhat_ev.copy()
        for i_outcome, outcome in enumerate(data['Xshortdims']['outcome']):
            for i_pot, pot in enumerate(data['Xshortdims']['pot']):
                yhat = np.squeeze(apply_fit(fit_param, np.expand_dims(data['Xshort'][i_outcome, i_pot, :, :], axis=0)))
                yhat_ev[i_outcome, i_pot, :] = np.mean(yhat, axis=0)

                y_ev[i_outcome, i_pot, :] = np.mean(data['Yshort'][outcome][i_pot], axis=0)

        emoev[stimid] = dict(
            ev_potoutcome=dict(
                model=yhat_ev,
                empir=y_ev,
            ),
            ev_outcome=dict(
                model=np.mean(yhat_ev, axis=1),
                empir=np.mean(y_ev, axis=1),
            )
        )

    stimid_list = list()
    alldeltas_model, alldeltas_empir = list(), list()
    allintenities_model, allintenities_empir = list(), list()
    for stimid, playerevs in emoev.items():
        if stimid != 'generic':
            stimid_list.append(stimid)

            deltas_model_outcome = playerevs['ev_outcome']['model'] - emoev['generic']['ev_outcome']['model']
            deltas_empir_outcome = playerevs['ev_outcome']['empir'] - emoev['generic']['ev_outcome']['empir']

            alldeltas_model.extend(deltas_model_outcome.flatten().tolist())
            alldeltas_empir.extend(deltas_empir_outcome.flatten().tolist())

            allintenities_model.extend(playerevs['ev_potoutcome']['model'].flatten().tolist())
            allintenities_empir.extend(playerevs['ev_potoutcome']['empir'].flatten().tolist())

    res_overall = dict(
        specific_test_deltas_byoutcome=concordance_corr_(alldeltas_model, alldeltas_empir),
        specific_test_absinten_bypotoutcome=concordance_corr_(allintenities_model, allintenities_empir),
    )

    res_data = dict()
    if withdata:
        model_deltas_list, empir_deltas_list = list(), list()
        model_ev_outcomepot_list, empir_ev_outcomepot_list = list(), list()
        for stimid, playerevs in emoev.items():
            if stimid != 'generic':
                model_ev_outcomepot_list.append(playerevs['ev_potoutcome']['model'])
                empir_ev_outcomepot_list.append(playerevs['ev_potoutcome']['empir'])

                model_deltas_list.append(playerevs['ev_outcome']['model'] - emoev['generic']['ev_outcome']['model'])
                empir_deltas_list.append(playerevs['ev_outcome']['empir'] - emoev['generic']['ev_outcome']['empir'])

        res_data['deltas'] = dict(
            model=np.stack(model_deltas_list, axis=0),
            empir=np.stack(empir_deltas_list, axis=0),
            dims=dict(stimid=stimid_list, outcome=datatest[stimid_list[0]]['Xshortdims']['outcome'], emotion=datatest[stimid_list[0]]['Yshortdims']['emotion']),
        )
        res_data['test_outcomepot'] = dict(
            model=np.stack(model_ev_outcomepot_list, axis=0),
            empir=np.stack(empir_ev_outcomepot_list, axis=0),
            dims=dict(stimid=stimid_list, outcome=datatest[stimid_list[0]]['Xshortdims']['outcome'], pot=datatest[stimid_list[0]]['Xshortdims']['pot'], emotion=datatest[stimid_list[0]]['Yshortdims']['emotion']),
        )

    return res_overall, res_data


def print_fit_summary(res_stats=None, res_stats_by_iter_df=None, res_data=None, optparam=None, data_dims=None, fit_param_temp=None, istep=0, outpath=None, plt=None):

    from webpypl_plotfun import printFigList
    from matplotlib.gridspec import GridSpec
    import numpy as np

    outcomes = ['CC', 'CD', 'DC', 'DD']
    emotions = data_dims['Y']['emotion']

    l1_scale = round(optparam['l1_scale']) if 'l1_scale' in optparam else 'NA'

    plt.close('all')
    figsout = list()

    nrows, ncols = 3, 3
    fig = plt.figure(figsize=(5 * ncols, 5 * nrows), constrained_layout=True, facecolor='white')
    gs = GridSpec(nrows, ncols, figure=fig)
    axs = dict()

    res_stats.keys()
    axid = 'deltas'
    axs[axid] = fig.add_subplot(gs[0, 0])
    axs[axid].scatter(res_data['deltas']['model'].flatten(), res_data['deltas']['empir'].flatten(), marker='o', facecolors='none', color='k', alpha=0.3)
    axs[axid].plot([-1, 1], [-1, 1], 'k--')
    axs[axid].set_xlim([-0.3, 0.3])
    axs[axid].set_ylim([-0.3, 0.3])
    axs[axid].set_aspect(aspect=1.0)
    axs[axid].set_xlabel('model deltas')
    axs[axid].set_ylabel('empir deltas')
    axs[axid].set_title(f"Deltas CCC : {res_stats['specific_test_deltas_byoutcome']:0.4f}\nniter: {istep}\nVar ratio: {np.var(res_data['deltas']['model'].flatten())/np.var(res_data['deltas']['empir'].flatten()):0.4f}")

    axid = 'deltas_ccc_lp'
    axs[axid] = fig.add_subplot(gs[0, 1])
    axs[axid].scatter(res_stats_by_iter_df['score'], res_stats_by_iter_df['deltas_ccc'])
    axs[axid].axhline(y=0.0, color='k', linestyle='-')
    axs[axid].axhline(y=0.4, color='grey', linestyle='--')
    axs[axid].set_xlabel('score')
    axs[axid].set_ylabel('concordance')
    # axs[axid].set_title(f"Deltas CCC : {res_stats['specific_test_deltas_byoutcome']:0.4f}\nTrain  CCC : {res_stats['train']['byoutcomepot']['concordance']:0.4f}\nTest   CCC : {res_stats['test']['byoutcomepot']['concordance']:0.4f}\n$\Delta$ concordance")

    axid = 'deltas_ccc_iter'
    axs[axid] = fig.add_subplot(gs[0, 2])
    axs[axid].plot(res_stats_by_iter_df['iiter'], res_stats_by_iter_df['deltas_ccc'])
    axs[axid].axhline(y=0.0, color='k', linestyle='-')
    axs[axid].axhline(y=0.4, color='grey', linestyle='--')
    axs[axid].set_xlabel('iter')
    axs[axid].set_ylabel('concordance')
    axs[axid].set_title('$\Delta$ concordance')

    axid = 'intens'
    axs[axid] = fig.add_subplot(gs[1, 0])
    for i_outcome, outcome in enumerate(outcomes):
        yhat_ = res_data['test_outcomepot']['model'][:, i_outcome, :, :].flatten()  # <20 specific_player, 4 outcome, 8 pot, 20 emotion>
        yemp_ = res_data['test_outcomepot']['empir'][:, i_outcome, :, :].flatten()  # <20 specific_player, 4 outcome, 8 pot, 20 emotion>
        color_ = ['green', 'blue', 'red', 'black'][i_outcome]
        axs[axid].scatter(yhat_, yemp_, marker='o', facecolors='none', color=color_, alpha=0.3)
    axs[axid].plot([-1, 1], [-1, 1], 'k--')
    axs[axid].set_xlim([0., 1.])
    axs[axid].set_ylim([0., 1.])
    axs[axid].set_aspect(aspect=1.0)
    axs[axid].set_xlabel('model $E[emo \mid a_1, a_2, pot]$')
    axs[axid].set_ylabel('empir $E[emo \mid a_1, a_2, pot]$')

    axid = 'intens_ccc_lp'
    axs[axid] = fig.add_subplot(gs[1, 1])
    axs[axid].scatter(res_stats_by_iter_df['score'], res_stats_by_iter_df['test_ccc'])
    axs[axid].set_ylim([0.0, 1.0])
    axs[axid].set_xlabel('score')
    axs[axid].set_ylabel('test_ccc')
    axs[axid].set_title('specific players concordance \n(absolute intensity)')

    axid = 'intens_ccc_iter'
    axs[axid] = fig.add_subplot(gs[1, 2])
    axs[axid].plot(res_stats_by_iter_df['iiter'], res_stats_by_iter_df['test_ccc'])
    axs[axid].set_ylim([0.0, 1.0])
    axs[axid].set_xlabel('iter')
    axs[axid].set_ylabel('test_ccc')
    axs[axid].set_title('specific players concordance \n(absolute intensity)')

    axid = 'betas'
    axs[axid] = fig.add_subplot(gs[2, :])
    vals_ = np.sort(np.abs(fit_param_temp['A'].numpy().flatten()))
    axs[axid].plot(range(vals_.size), vals_)
    axs[axid].axhline(y=0, color='k', linewidth=1)
    axs[axid].set_title('abs A')

    figpath_byiter = outpath / f"pytorch_trace_l1-{l1_scale}_3byiter" / f"pytorch_trace_l1-{l1_scale}_niter-{istep}.png"
    figpath_current = outpath / f"pytorch_trace_l1-{l1_scale}_1summary.png"

    figsout.append((figpath_current, fig, True, figpath_byiter))

    figpath_root = outpath.parents[2] / 'figs_collected' / f"{outpath.parents[1].name}-{outpath.parents[0].name.split('_T-')[-1]}.png"
    figsout.append((figpath_root, fig, False, ))

    plt.close(fig)

    ################

    nrows, ncols = 6, 4
    fig, axs_ = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True, facecolor='white')
    axs = axs_.flatten()

    res_data['test_outcomepot']['model'].shape
    res_data['test_outcomepot']['empir'].shape
    for i_outcome, outcome in enumerate(outcomes):
        axid = (i_outcome,)
        color_ = ['green', 'blue', 'red', 'black'][i_outcome]
        yhat_ = res_data['test_outcomepot']['model'][:, i_outcome, :, :].flatten()  # <20 specific_player, 4 outcome, 8 pot, 20 emotion>
        yemp_ = res_data['test_outcomepot']['empir'][:, i_outcome, :, :].flatten()  # <20 specific_player, 4 outcome, 8 pot, 20 emotion>
        axs[axid].scatter(yhat_, yemp_, marker='o', facecolors='none', color=color_, alpha=0.3)
        axs[axid].plot([-1, 1], [-1, 1], 'k--')
        axs[axid].set_xlim([0., 1.])
        axs[axid].set_ylim([0., 1.])
        axs[axid].set_aspect(aspect=1.0)
        axs[axid].set_xlabel('model $E[emo \mid a_1, a_2, pot]$')
        axs[axid].set_ylabel('empir $E[emo \mid a_1, a_2, pot]$')

    for i_emotion, emotion in enumerate(emotions):
        axid = (i_emotion + 4,)
        for i_outcome, outcome in enumerate(outcomes):
            color_ = ['green', 'blue', 'red', 'black'][i_outcome]
            yhat_ = res_data['test_outcomepot']['model'][:, i_outcome, :, i_emotion].flatten()  # <20 specific_player, 4 outcome, 8 pot, 20 emotion>
            yemp_ = res_data['test_outcomepot']['empir'][:, i_outcome, :, i_emotion].flatten()  # <20 specific_player, 4 outcome, 8 pot, 20 emotion>
            axs[axid].scatter(yhat_, yemp_, marker='o', facecolors='none', color=color_, alpha=0.5)
        axs[axid].plot([-1, 1], [-1, 1], 'k--')
        axs[axid].set_xlim([0., 1.])
        axs[axid].set_ylim([0., 1.])
        axs[axid].set_aspect(aspect=1.0)
        axs[axid].set_xlabel('model $E[emo \mid a_1, a_2, pot]$')
        axs[axid].set_ylabel('empir $E[emo \mid a_1, a_2, pot]$')
        axs[axid].set_title(emotion)

    # fig.suptitle(f"Deltas CCC : {res_stats['deltas']['byoutcome']['concordance']:0.4f}\nTrain  CCC : {res_stats['train']['byoutcomepot']['concordance']:0.4f}\nTest   CCC : {res_stats['test']['byoutcomepot']['concordance']:0.4f}, \nniter: {istep}")

    figpath_current = outpath / f"pytorch_trace_l1-{l1_scale}_2emos.png"

    figsout.append((figpath_current, fig, False))

    _ = printFigList(figsout, plt)

    plt.close('all')


# %%


def run(datatrain=None, datatest=None, model_param=None, optimization_param=None, outpath=None, trackprogress=True):

    import matplotlib
    import matplotlib.pyplot as plt
    import pickle
    import pandas as pd
    from pathlib import Path
    import numpy as np

    isInteractive = False
    try:
        if __IPYTHON__:  # type: ignore
            get_ipython().run_line_magic('matplotlib', 'inline')  # type: ignore
            get_ipython().run_line_magic('load_ext', 'autoreload')  # type: ignore
            get_ipython().run_line_magic('autoreload', '2')  # type: ignore
            isInteractive = True
    except NameError:
        isInteractive = False

    if not isInteractive:
        matplotlib.use('pdf')

    # if progresstrack is None:
    #     progresstrack = dict()
    # prog_track = progresstrack.get('track', False)

    l1_scale = model_param['laplace_scale']
    logit_k = model_param['k']

    niter = optimization_param['iter']
    seed = optimization_param.get('seed', None)

    # %%

    # datatrain.keys()
    # datatrain['generic'].keys()

    """
    l1_scale = 260
    logit_k = 0.5
    """
    optparam = {'l1_scale': l1_scale, 'kind': 'LASSO', 'logit_k': logit_k}
    model_trainer = get_predictions(datatrain, optparam, seed=seed)

    # %%

    """
    niter = 200
    """
    for _ in range(niter + 1):
        model_trainer.step_optimizer()

        if model_trainer.istep % 10 == 0 and model_trainer.istep > 0:
            model_trainer.logger()

        if trackprogress:

            if (model_trainer.istep % 100 == 0 and model_trainer.istep > 0) or model_trainer.istep == 10:
                print(f"{model_trainer.istep}")

                modelfit_log = model_trainer.get_fit_logger()

                res_stats_by_iter = list()
                for modelfit_iter in modelfit_log:
                    res_stats, _ = calc_fit(modelfit_iter['param'], datatest, withdata=False)
                    res_stats_by_iter.append(dict(
                        iiter=modelfit_iter['iiter'],
                        score=modelfit_iter['score'],
                        # train_mse=resstats_['train']['byoutcomepot']['mse'],
                        # test_mse=resstats_['test']['byoutcomepot']['mse'],
                        test_ccc=res_stats['specific_test_absinten_bypotoutcome'],
                        # deltas_mse=resstats_['deltas']['byoutcome']['mse'],
                        deltas_ccc=res_stats['specific_test_deltas_byoutcome'],
                    ))
                res_stats_by_iter_df = pd.DataFrame(res_stats_by_iter)
                print(res_stats_by_iter_df.iloc[-1, :])

                fit_param_temp = model_trainer.get_current_param()
                res_stats, res_data = calc_fit(modelfit_iter['param'], datatest, withdata=True)

                data_dims = dict(Y=dict(emotion=datatest['generic']['Yshortdims']['emotion']))
                print_fit_summary(res_stats=res_stats, res_stats_by_iter_df=res_stats_by_iter_df, res_data=res_data, optparam=optparam, data_dims=data_dims, fit_param_temp=fit_param_temp, istep=model_trainer.istep, outpath=outpath, plt=plt)

                # print(f"Train  CCC : {res_stats['train']['byoutcomepot']['concordance']}")
                # print(f"Test   CCC : {res_stats['test']['byoutcomepot']['concordance']}")
                # print(f"Deltas CCC : {res_stats['deltas']['byoutcome']['concordance']}")

                for k_, v_ in fit_param_temp.items():
                    print(f"{k_} ({v_.size()}): {v_.mean():0.4f}")

    # %%

    learned_param = model_trainer.get_current_param()
    res_stats_final, res_data_final = calc_fit(learned_param, datatest, withdata=True)

    model_trainer.logger()
    modelfit_log = model_trainer.get_fit_logger()
    res_stats_by_iter = list()
    for modelfit_iter in modelfit_log:
        res_stats, _ = calc_fit(modelfit_iter['param'], datatest, withdata=False)
        res_stats_by_iter.append(dict(
            iiter=modelfit_iter['iiter'],
            score=modelfit_iter['score'],
            test_ccc=res_stats['specific_test_absinten_bypotoutcome'],
            deltas_ccc=res_stats['specific_test_deltas_byoutcome'],
        ))
    res_stats_by_iter_df = pd.DataFrame(res_stats_by_iter)

    data_dims = dict(Y=dict(emotion=datatest['generic']['Yshortdims']['emotion']))
    print_fit_summary(res_stats=res_stats_final, res_stats_by_iter_df=res_stats_by_iter_df, res_data=res_data_final, optparam=optparam, data_dims=data_dims, fit_param_temp=learned_param, istep=model_trainer.istep, outpath=outpath, plt=plt)

    final_res = dict(
        learned_param=learned_param,
        stats=res_stats_final,
        appliedfit=res_data_final,
        progressdf=res_stats_by_iter_df,
    )
    return final_res


# %%
