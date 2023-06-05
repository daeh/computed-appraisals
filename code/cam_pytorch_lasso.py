#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_pytorch_lasso.py
"""


import torch
import numpy as np
import random


def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If using cuda
    # torch.cuda.manual_seed_all(seed)
    # If running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    # os.environ["PYTHONHASHSEED"] = str(seed)
    torch.set_num_threads(1)


def get_predictions(traindatain, opt_param):

    import math
    from torch import nn
    from torch.nn import functional as F
    from torch.distributions import Normal, Laplace
    from torch.optim import Adam
    from torch.distributions.transforms import ComposeTransform, AffineTransform, SigmoidTransform

    def inverse_soft_plus(x):
        return math.log(math.exp(x) - 1)

    def inverse_soft_plus_torch(x):
        return torch.log(torch.exp(x) - 1)

    class Lin(nn.Module):
        def __init__(self, n_emotions, n_features, opt_dict=None):
            super().__init__()

            self.data = None

            ### NB no prior on sigma and b ###

            ### init around zero ###
            self.b = nn.Parameter(0.01 * torch.randn(n_emotions))
            ### init around 1 ###
            self._sigma_emotions = nn.Parameter(inverse_soft_plus(1.0) * torch.ones(n_emotions) + 0.01 * torch.randn(n_emotions))

            self.logit_k_param = opt_dict.get('logit_k', 0.4)
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
            return self._logit_k

        @property
        def invaffine_logistic(self):
            return ComposeTransform([AffineTransform(loc=torch.Tensor([0.0]), scale=self.logit_k), SigmoidTransform()])

        def score_single_dataset(self, x, y, jy):
            ### x size : <(n pots * 4 outcomes) conditions, n samples per condition, n appraisal features>
            ### mu size : <(n pots * 4 outcomes) conditions, n samples per condition, n emotions> -- predicted vector of emotion intensities corresponding to each sample
            ### assumes that the first dimension of x corresponseds to stim id [1,2,3,4,...,(n pots * 4 outcomes)]
            ### i.e. maps the jy id to an x row
            mu_logit = (x[:, :, None, :] * self.A[None, None, :, :]).sum(dim=3) + self.b[None, None, :]

            ### transform predicted values into [0,1] space of empirical data
            mu_logistic = self.invaffine_logistic(mu_logit)

            dist = Normal(mu_logistic[jy], self.sigma_emotions[None, None, :])

            ### calculate the probability density of observing \vec{e_i} in the mixture
            ### i.e. score the 20d vector y[i,None,0:19] in the Gaussian over mu_logistic[jy[i],j,0:19]
            ### where j is an index of webppl samples and jy[i] gives the stimulus ID of \vec{e_i}
            ### multiply the probability densities of each dimension of the 20d empirical emotion vector to give a probability for the empirical vector in each webppl sample corresponding to that stimulus (e.g. score might be 4432 empirical observations x 375 webppl samples)
            score = dist.log_prob(y[:, None, :]).sum(dim=2)

            return score

        def score(self):
            loglik_data_list = [self.score_single_dataset(data['X'], data['Y'], data['Jy']) for data in self.data]

            ### stack (observations x webppl samples) lp across datasets (now n_total_empirical_observations x n_webpplsamples)
            ### where each element is the lp of observing a given 20d empirical vector in a given 20d gaussian
            loglik_data_stacked = torch.cat(loglik_data_list, dim=0)

            ### loglik_data_stacked.logsumexp(dim=1) <- for each empirical vector, sum the probabilities (not the log probabilities) of observing that vector in all the webppl normals
            ### divide the total by the number of webppl samples to normalize
            ### take the mean of all the empirical obserations to yield the probability of observing the empirical data in the mixture
            ### taking the mean rather than the sum of the logprobs allows for comparision between datasets with different numbers of empirical observations

            ### loglik_data_stacked <n_total_empirical_observations, n_wppl_samples>
            ### loglik_data_stacked.logsumexp(dim=1) <n_total_empirical_observations>, loglik_data <1>
            loglik_data = (loglik_data_stacked.logsumexp(dim=1) - math.log(loglik_data_stacked.size(1))).mean()

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
            }.items():
                dict_out[key_] = torch.empty_like(var_, requires_grad=False).copy_(var_).detach()
            return dict_out

    class ModelTrainer():
        def __init__(self, datain):

            firstkey = list(datain.keys())[0]

            ### initialize on columns of Y and X, which should be consistant ###
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

    return ModelTrainer(traindatain)


def apply_fit(fit_param, X_in):

    def invaffine_logistic(logit_k):
        from torch.distributions.transforms import ComposeTransform, AffineTransform, SigmoidTransform
        return ComposeTransform([AffineTransform(loc=torch.Tensor([0.0]), scale=logit_k), SigmoidTransform()])

    invaffine_logistic_transform = invaffine_logistic(fit_param['logit_k'])

    x = torch.Tensor(X_in)

    A_ = fit_param['A'][None, None, :, :]

    x_ = x[:, :, None, :]
    b_ = fit_param['b'][None, None, :]

    mu_logit = (x_ * A_).sum(dim=3) + b_

    mu_logistic = invaffine_logistic_transform(mu_logit)

    return mu_logistic.detach().numpy()


def calc_fit(fit_param, datatest, withdata=False):
    from cam_utils import concordance_corr_

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


def print_fit_summary(res_stats=None, res_stats_by_iter_df=None, res_data=None, optparam=None, data_dims=None, fit_param_temp=None, istep=0, outpath=None):

    import matplotlib
    import matplotlib.pyplot as plt
    from cam_plot_utils import printFigList
    from matplotlib.gridspec import GridSpec

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

    axid = 'deltas_ccc_iter'
    axs[axid] = fig.add_subplot(gs[0, 2])
    axs[axid].plot(res_stats_by_iter_df['iiter'], res_stats_by_iter_df['deltas_ccc'])
    axs[axid].axhline(y=0.0, color='k', linestyle='-')
    axs[axid].axhline(y=0.4, color='grey', linestyle='--')
    axs[axid].set_xlabel('iter')
    axs[axid].set_ylabel('concordance')
    axs[axid].set_title(f"$\Delta$ concordance: {res_stats['specific_test_deltas_byoutcome']:0.4f}")

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
    axs[axid].set_title(f"specific players CCC: {res_stats['specific_test_absinten_bypotoutcome']:0.4f}")

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
    axs[axid].axhline(y=0.9, color='grey', linestyle='--')
    axs[axid].axhline(y=0.8, color='grey', linestyle='--')
    axs[axid].set_xlabel('iter')
    axs[axid].set_ylabel('test_ccc')
    axs[axid].set_title(f"specific players concordance: {res_stats['specific_test_absinten_bypotoutcome']:0.4f} \n(absolute intensity)")

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

    figpath_current = outpath / f"pytorch_trace_l1-{l1_scale}_2emos.png"

    figsout.append((figpath_current, fig, False))

    _ = printFigList(figsout, plt)

    plt.close('all')


def run(datatrain=None, datatest=None, model_param=None, optimization_param=None, outpath=None, trackprogress=True):

    import pandas as pd

    l1_scale = model_param['laplace_scale']
    logit_k = model_param['k']

    niter = optimization_param['iter']
    seed = optimization_param.get('seed', None)

    set_seed(seed)

    # %%

    optparam = {'l1_scale': l1_scale, 'kind': 'LASSO', 'logit_k': logit_k}
    model_trainer = get_predictions(datatrain, optparam)

    # %%

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
                        test_ccc=res_stats['specific_test_absinten_bypotoutcome'],
                        deltas_ccc=res_stats['specific_test_deltas_byoutcome'],
                    ))
                res_stats_by_iter_df = pd.DataFrame(res_stats_by_iter)
                print(res_stats_by_iter_df.iloc[-1, :])

                fit_param_temp = model_trainer.get_current_param()
                res_stats, res_data = calc_fit(modelfit_iter['param'], datatest, withdata=True)

                data_dims = dict(Y=dict(emotion=datatest['generic']['Yshortdims']['emotion']))
                print_fit_summary(res_stats=res_stats, res_stats_by_iter_df=res_stats_by_iter_df, res_data=res_data, optparam=optparam, data_dims=data_dims, fit_param_temp=fit_param_temp, istep=model_trainer.istep, outpath=outpath)

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
    res_stats_by_iter_df.drop_duplicates(subset=['iiter'])

    data_dims = dict(Y=dict(emotion=datatest['generic']['Yshortdims']['emotion']))
    print_fit_summary(res_stats=res_stats_final, res_stats_by_iter_df=res_stats_by_iter_df, res_data=res_data_final, optparam=optparam, data_dims=data_dims, fit_param_temp=learned_param, istep=model_trainer.istep, outpath=outpath)

    final_res = dict(
        learned_param=learned_param,
        stats=res_stats_final,
        appliedfit=res_data_final,
        progressdf=res_stats_by_iter_df,
    )
    return final_res
