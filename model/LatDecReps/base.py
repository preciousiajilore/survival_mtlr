from abc import abstractmethod
from collections import OrderedDict
import warnings
import pandas as pd
from tqdm import trange
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from SurvivalEVAL.Evaluations.util import NumericArrayLike

from model.loss import OrthoNets, OrthoReps, IPM, Likelihood
from model.utils import build_sequential_nn
from utils.util_survival import extract_survival


class SurvivalBaseLDR(nn.Module):
    """
    Base class for survival models which generates survival distributions via
    learning latent decomposed representations (LDR).

    It first uses three independent NN to generate three hidden representation:
    1. one for event distribution only (epsilon net),
    2. one for censoring distribution only (gamma net),
    3. and the last one for the shared representation (kappa net).

    Then, it uses four independent NN to generate the logits/outputs:
    1. one for event time logits, using epsilon net only,
    2. one for event time logits, using epsilon net and kappa net,
    3. one for censoring time logits, using gamma net only,
    4. one for censoring time logits, using gamma net and kappa net.

    Furthermore, compared to SurvivalBase and SurvivalBase2B, the SurvivalBaseDR has additional outputs
    for the hidden representations of epsilon, gamma, and kappa nets.
    """

    def __init__(
            self,
            n_features: int,
            rep_dims: list,
            event_dims: list,
            censor_dims: list,
            output_size_event: int,
            output_size_censor: int,
            norm: bool,
            activation: str,
            dropout: float,
            ipm: str,
            alpha: float,
            beta: float,
    ):
        super().__init__()
        self.in_features = n_features
        assert len(rep_dims) > 0, "rep_dims should have at least one element"
        self._rep_dims = rep_dims
        self._event_dims = event_dims
        self._censor_dims = censor_dims
        self.dropout = dropout
        self.norm = norm
        self.activation = activation
        self.alpha = alpha
        self.beta = beta
        self.loss = None
        self.orthog_reg = OrthoNets(n_features, alpha)
        # self.orthog_reg = OrthoReps(alpha)
        self.balance_reg = IPM(ipm, beta)
        self.output_size_event = output_size_event
        self.output_size_censor = output_size_censor
        self.output_size = output_size_event + output_size_censor

        # representation nets
        self.epsilon_net = self._build_rep_net()
        self.gamma_net = self._build_rep_net()
        self.kappa_net = self._build_rep_net()

        # output nets
        self.event_epsilon = self._build_dist_net(self._rep_dims[-1], self._event_dims, self._output_size_event)
        self.event_epsilon_kappa = self._build_dist_net(2 * self._rep_dims[-1], self._event_dims, self._output_size_event)
        self.censor_gamma = self._build_dist_net(self._rep_dims[-1], self._censor_dims, self._output_size_censor)
        self.censor_gamma_kappa = self._build_dist_net(2 * self._rep_dims[-1], self._censor_dims, self._output_size_censor)

    @property
    def rep_dims(self):
        return self._rep_dims

    @rep_dims.setter
    def rep_dims(self, value):
        self._rep_dims = value

    @property
    def event_dims(self):
        return self._event_dim

    @event_dims.setter
    def event_dims(self, value):
        self._event_dim = value

    @property
    def censor_dims(self):
        return self._censor_dim

    @censor_dims.setter
    def censor_dims(self, value):
        self._censor_dim = value

    @property
    def output_size_event(self):
        return self._output_size_event

    @output_size_event.setter
    def output_size_event(self, value):
        self._output_size_event = value

    @property
    def output_size_censor(self):
        return self._output_size_censor

    @output_size_censor.setter
    def output_size_censor(self, value):
        self._output_size_censor = value

    @property
    def output_size(self):
        return self._output_size

    @output_size.setter
    def output_size(self, value):
        self._output_size = value

    @abstractmethod
    def predict_survival(self, x):
        pass

    def predict_cdf(self, x):
        return 1 - self.predict_survival(x)

    @abstractmethod
    def predict_time(self, x, pred_type='mean'):
        pass

    def _build_rep_net(
            self,
    ):
        """
        Build a simple feedforward MLP with the specified [input, [hidden], and output] dimensions.
        """
        layers = OrderedDict()
        in_dim = self.in_features
        hid_dim = self.rep_dims[0]
        for i in range(len(self.rep_dims)):
            layers[f'linear{i}'] = nn.Linear(in_dim, hid_dim)

            if self.norm:
                layers[f'batchnorm{i}'] = nn.BatchNorm1d(hid_dim)
            layers[f'activation{i}'] = getattr(nn, self.activation)()

            if self.dropout is not None and self.dropout > 0:
                layers[f'dropout{i}']=nn.Dropout(self.dropout)

            in_dim = hid_dim
            hid_dim = self.rep_dims[i + 1] if i + 1 < len(self.rep_dims) else None
        return nn.Sequential(layers)

    @abstractmethod
    def _build_dist_net(self, in_dim, hidden_dims, out_dim=1):
        pass

    def forward(self, x):
        epsilon = self.epsilon_net(x)
        gamma = self.gamma_net(x)
        kappa = self.kappa_net(x)

        # normalize the representation
        epsilon = F.normalize(epsilon, p=2, dim=1)
        gamma = F.normalize(gamma, p=2, dim=1)
        kappa = F.normalize(kappa, p=2, dim=1)

        event_eps = self.event_epsilon(epsilon)
        event_eps_kap = self.event_epsilon_kappa(torch.cat([epsilon, kappa], dim=1))
        censor_gam = self.censor_gamma(gamma)
        censor_gam_kap = self.censor_gamma_kappa(torch.cat([gamma, kappa], dim=1))

        return event_eps, event_eps_kap, censor_gam, censor_gam_kap, epsilon, gamma, kappa

    def forward_e(self, x):
        epsilon = self.epsilon_net(x)
        kappa = self.kappa_net(x)

        # normalize the representation
        epsilon = F.normalize(epsilon, p=2, dim=1)
        kappa = F.normalize(kappa, p=2, dim=1)

        event_eps_kap = self.event_epsilon_kappa(torch.cat([epsilon, kappa], dim=1))

        return event_eps_kap

    def forward_c(self, x):
        gamma = self.gamma_net(x)
        kappa = self.kappa_net(x)

        # normalize the representation
        gamma = F.normalize(gamma, p=2, dim=1)
        kappa = F.normalize(kappa, p=2, dim=1)

        censor_gam_kap = self.censor_gamma_kappa(torch.cat([gamma, kappa], dim=1))

        return censor_gam_kap

    def reset_parameters(self):
        # reset the parameters of all the representation nets and output nets
        for layer in self.epsilon_net.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.gamma_net.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.kappa_net.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.event_epsilon.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.event_epsilon_kappa.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.censor_gamma.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.censor_gamma_kappa.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class ContinuousSurvivalBaseLDR(SurvivalBaseLDR):
    """
    Base class for continuous survival models which generates continuous survival distributions via
    learning latent decomposed representations (LDR).
    """

    def __init__(
            self,
            n_features: int,
            rep_dims: list,
            event_dims: list,
            censor_dims: list,
            norm: bool,
            activation: str,
            dropout: float,
            ipm: str,
            alpha: float,
            beta: float,
            time_grids_e: NumericArrayLike = None,
            time_grids_c: NumericArrayLike = None
    ):
        super().__init__(
            n_features=n_features,
            rep_dims=rep_dims,
            event_dims=event_dims,
            censor_dims=censor_dims,
            output_size_event=1,
            output_size_censor=1,
            norm=norm,
            activation=activation,
            dropout=dropout,
            ipm=ipm,
            alpha=alpha,
            beta=beta
        )
        if time_grids_e is None or time_grids_c is None:
            warnings.warn("'time_grids' is not provided, the time grids will be generated using the time points "
                          "in the training data.")
        self.t_grids_e = time_grids_e
        self.t_grids_c = time_grids_c
        self.t_grids = self.t_grids_e
        self.time_bins_e = None
        self.time_bins_c = None
        self.time_bins = self.time_bins_e
        self.loss = Likelihood(reduction="mean")

    def _build_dist_net(self, in_dim, hidden_dim, out_dim=1):
        if not hidden_dim:
            # if hidden_size is empty, then the only layer is linear
            layers = [nn.Linear(in_dim, out_dim)]
        else:
            layers = build_sequential_nn(in_dim, hidden_dim, self.norm, self.activation, self.dropout)
            layers.append(nn.Linear(hidden_dim[-1], out_dim))
        return nn.Sequential(*layers)


    def predict_pdf(self, x, target='e'):
        self.eval()
        with torch.no_grad():
            if target == 'e':
                t_grids = self.t_grids_e.repeat(x.shape[0], 1)
            elif target == 'c':
                t_grids = self.t_grids_c.repeat(x.shape[0], 1)
            else:
                raise ValueError(f"Invalid target: {target}")
            return self._predict_pdf(x, t_grids, target)

    def predict_survival(self, x, target='e'):
        self.eval()
        with torch.no_grad():
            if target == 'e':
                t_grids = self.t_grids_e.repeat(x.shape[0], 1)
            elif target == 'c':
                t_grids = self.t_grids_c.repeat(x.shape[0], 1)
            else:
                raise ValueError(f"Invalid target: {target}")
            return self._predict_survival(x, t_grids, target)

    def predict_hazard(self, x, target='e'):
        self.eval()
        with torch.no_grad():
            if target == 'e':
                t_grids = self.t_grids_e.repeat(x.shape[0], 1)
            elif target == 'c':
                t_grids = self.t_grids_c.repeat(x.shape[0], 1)
            else:
                raise ValueError(f"Invalid target: {target}")
            return self._predict_hazard(x, t_grids, target)

    def predict_cum_hazard(self, x, target='e'):
        self.eval()
        with torch.no_grad():
            if target == 'e':
                t_grids = self.t_grids_e.repeat(x.shape[0], 1)
            elif target == 'c':
                t_grids = self.t_grids_c.repeat(x.shape[0], 1)
            else:
                raise ValueError(f"Invalid target: {target}")
            return self._predict_cum_hazard(x, t_grids, target)

    def predict_quantiles(self, x, quantiles=None, target='e'):
        self.eval()
        with torch.no_grad():
            return self._predict_quantiles(x, quantiles, target)

    def predict_time(self, x, target='e', pred_type='mean'):
        raise NotImplementedError("Time prediction is not implemented for continuous survival models.")

    @abstractmethod
    def _predict_pdf(self, x, times, target='all'):
        pass

    @abstractmethod
    def _predict_survival(self, x, times, target='all'):
        pass

    @abstractmethod
    def _predict_hazard(self, x, times, target='all'):
        pass

    @abstractmethod
    def _predict_cum_hazard(self, x, times, target='all'):
        pass

    @abstractmethod
    def _predict_quantiles(self, x, quantiles, target='all'):
        pass

    def _predict_risks(self, x, times, target='all'):
        if target == 'all':
            output_e1, output_e2, output_c1, output_c2, epsilon, gamma, kappa = self.forward(x)
            risks_e1, risks_e2, risks_c1, risks_c2 = torch.exp(output_e1), torch.exp(output_e2), torch.exp(
                output_c1), torch.exp(output_c2)
            ndim = times.dim()
            if ndim == 1:
                risks_e1 = risks_e1.squeeze()
                risks_e2 = risks_e2.squeeze()
                risks_c1 = risks_c1.squeeze()
                risks_c2 = risks_c2.squeeze()
            elif ndim == 2:
                pass
            else:
                raise ValueError(f"Invalid dimension: {ndim}")

            return risks_e1, risks_e2, risks_c1, risks_c2, epsilon, gamma, kappa
        elif target == 'both':
            _, output_e2, _, output_c2, _, _, _ = self.forward(x)
            risks_e2, risks_c2 = torch.exp(output_e2), torch.exp(output_c2)
            ndim = times.dim()
            if ndim == 1:
                risks_e2 = risks_e2.squeeze()
                risks_c2 = risks_c2.squeeze()
            elif ndim == 2:
                pass
            else:
                raise ValueError(f"Invalid dimension: {ndim}")

            return risks_e2, risks_c2
        elif target == 'e':
            output_e2 = self.forward_e(x)
            risks_e2 = torch.exp(output_e2)
            ndim = times.dim()
            if ndim == 1:
                risks_e2 = risks_e2.squeeze()
            elif ndim == 2:
                pass
            else:
                raise ValueError(f"Invalid dimension: {ndim}")

            return risks_e2
        elif target == 'c':
            output_c2 = self.forward_c(x)
            risks_c2 = torch.exp(output_c2)
            ndim = times.dim()
            if ndim == 1:
                risks_c2 = risks_c2.squeeze()
            elif ndim == 2:
                pass
            else:
                raise ValueError(f"Invalid dimension: {ndim}")

            return risks_c2

    def fit(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            device: torch.device,
            batch_size: int,
            epochs: int,
            optimizer: str,
            lr: float,
            lr_min: float,
            weight_decay: float,
            early_stop: bool = True,
            patience: int = 50,
            fname: str = '',
            verbose: bool = True
    ):
        self.reset_parameters()
        self.to(device)
        self.orthog_reg.to(device)

        if self.t_grids_e is None or self.t_grids_c is None:
            self.t_grids = torch.tensor(train_df['time'], dtype=torch.float64).to(device).unique()
            self.t_grids = torch.cat([torch.tensor([0.], dtype=torch.float64).to(device), self.t_grids], dim=0)
            self.t_grids_e = self.t_grids
            self.t_grids_c = self.t_grids

        optimizer = getattr(torch.optim, optimizer)
        optim = optimizer((param for param in self.parameters() if param.requires_grad), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=lr_min)
        x_train, ye_train, yc_train, _, _ = extract_survival(train_df, self.time_bins_e, self.time_bins_c,
                                                                   include_censor_label=True)
        train_dataloader = DataLoader(TensorDataset(x_train, ye_train, yc_train),
                                      batch_size=batch_size, shuffle=True)
        if self.beta > 0:
            self.balance_reg.fit(ye_train[:, 0], ye_train[:, 1], device=device)

        if not val_df.empty:
            x_val, ye_val, yc_val, t_val, _ = extract_survival(val_df, self.time_bins_e, self.time_bins_c,
                                                               include_censor_label=True)
            x_val, ye_val, yc_val, t_val = x_val.to(device), ye_val.to(device), yc_val.to(device), t_val.to(device)

        best_loss = float('inf')
        best_ep = -1

        # training and evaluation
        prefix = f'Training w Early Stop on {device}' if early_stop else f'Training on {device} w/o Early Stop'
        pbar = trange(epochs, disable=not verbose, desc=prefix)
        for ep in pbar:
            # start training
            self.train()
            train_loss_ep = 0
            for xi, yei, yci in train_dataloader:
                xi, yei, yci = xi.to(device), yei.to(device), yci.to(device)

                ti = yei[:, 0]
                surv_e_eps, surv_e_eps_kap, surv_c_gam, surv_c_gam_kap, epsilon, gamma, kappa = self._predict_survival(xi, ti, 'all')
                pdf_e_eps, pdf_e_eps_kap, pdf_c_gam, pdf_c_gam_kap, _, _, _ = self._predict_pdf(xi, ti, 'all')
                # likelihood loss for event and censoring prediction
                loss = 0.25 * (self.loss(surv_e_eps, pdf_e_eps, yei) + self.loss(surv_e_eps_kap, pdf_e_eps_kap, yei) +
                        self.loss(surv_c_gam, pdf_c_gam, yci) + self.loss(surv_c_gam_kap, pdf_c_gam_kap, yci))
                loss += self.orthog_reg(self) if self.alpha > 0 else 0
                # loss += self.orthog_reg([epsilon, gamma, kappa]) if self.alpha > 0 else 0
                loss += self.balance_reg(epsilon, yci[:, 0], yci[:, 1], event=False) if self.beta > 0 else 0
                loss += self.balance_reg(gamma, yei[:, 0], yei[:, 1], event=True) if self.beta > 0 else 0

                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss_ep += loss.detach().item()

            scheduler.step()
            train_loss_ep /= len(train_dataloader)
            # evaluation
            self.eval()
            with torch.no_grad():
                postfix = f"Train loss = {train_loss_ep:.4f};"

                if early_stop and not val_df.empty:
                    surv_e_eps_val, surv_e_eps_kap_val, surv_c_gam_val, surv_c_gam_kap_val, eps_val, gam_val, kap_val = self._predict_survival(x_val, t_val, 'all')
                    pdf_e_eps_val, pdf_e_eps_kap_val, pdf_c_gam_val, pdf_c_gam_kap_val, _, _, _ = self._predict_pdf(x_val, t_val, 'all')

                    eval_ll = 0.25* (self.loss(surv_e_eps_val, pdf_e_eps_val, ye_val) +
                                 self.loss(surv_e_eps_kap_val, pdf_e_eps_kap_val, ye_val) +
                                 self.loss(surv_c_gam_val, pdf_c_gam_val, yc_val) +
                                 self.loss(surv_c_gam_kap_val, pdf_c_gam_kap_val, yc_val))
                    eval_orth = self.orthog_reg(self) if self.alpha > 0 else 0
                    # eval_orth = self.orthog_reg([eps_val, gam_val, kap_val]) if self.alpha > 0 else 0
                    eval_blc = self.balance_reg(eps_val, yc_val[:, 0], yc_val[:, 1], event=False) if self.beta > 0 else 0
                    eval_blc += self.balance_reg(gam_val, ye_val[:, 0], ye_val[:, 1], event=True) if self.beta > 0 else 0
                    postfix += f" Val nll = {eval_ll:.4f}, orth = {eval_orth:.4f}, blc = {eval_blc:.4f};"
                    eval_loss = eval_ll + eval_orth + eval_blc

                    if best_loss > eval_loss:
                        best_loss = eval_loss
                        best_ep = ep
                        torch.save({'model_state_dict': self.state_dict()}, fname + '.pth')
                    if (ep - best_ep) > patience:
                        postfix += f" Early stop at epoch {ep}. Best epoch is {best_ep}. Start testing..."
                        pbar.set_postfix_str(postfix)
                        break
                pbar.set_postfix_str(postfix)
        self.load_state_dict(torch.load(fname + '.pth')['model_state_dict']) if early_stop and not val_df.empty else None


class DiscreteSurvivalBaseLDR(SurvivalBaseLDR):
    """
    Base class for discrete survival models which generates discrete survival distributions via
    learning latent decomposed representations (LDR).
    """

    def __init__(
            self,
            n_features: int,
            rep_dims: list,
            event_dims: list,
            censor_dims: list,
            output_size_event: int,
            output_size_censor: int,
            norm: bool,
            activation: str,
            dropout: float,
            ipm: str,
            alpha: float,
            beta: float
    ):
        super().__init__(
            n_features=n_features,
            rep_dims=rep_dims,
            event_dims=event_dims,
            censor_dims=censor_dims,
            output_size_event=output_size_event,
            output_size_censor=output_size_censor,
            norm=norm,
            activation=activation,
            dropout=dropout,
            ipm=ipm,
            alpha=alpha,
            beta=beta
        )

    def _build_dist_net(self, in_dim, hidden_dims, out_dim=1):
        if not hidden_dims:
            # if hidden_size is empty, then the only layer is linear
            layers = [nn.Linear(in_dim, out_dim)]
        else:
            layers = build_sequential_nn(in_dim, hidden_dims, self.norm, self.activation, self.dropout)
            layers.append(nn.Linear(hidden_dims[-1], out_dim))
        return nn.Sequential(*layers)


    def predict_survival(self, x):
        pass

    def predict_time(self, x, pred_type='mean'):
        pass

    def fit(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            device: torch.device,
            batch_size: int,
            epochs: int,
            optimizer: str,
            lr: float,
            lr_min: float,
            weight_decay: float,
            early_stop: bool = True,
            patience: int = 50,
            fname: str = '',
            verbose: bool = True
    ):
        self.reset_parameters()
        self.to(device)
        self.orthog_reg.to(device)

        optimizer = getattr(torch.optim, optimizer)
        optim = optimizer((param for param in self.parameters() if param.requires_grad), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=lr_min)
        x_train, ye_train, yc_train, t_train, e_train = extract_survival(train_df, self.time_bins_e, self.time_bins_c,
                                                                         include_censor_label=True)
        train_dataloader = DataLoader(TensorDataset(x_train, ye_train, yc_train, t_train, e_train),
                                      batch_size=batch_size, shuffle=True)

        if self.beta > 0:
            self.balance_reg.fit(t_train, e_train, device=device)

        if not val_df.empty:
            x_val, ye_val, yc_val, t_val, e_val = extract_survival(val_df, self.time_bins_e, self.time_bins_c,
                                                           include_censor_label=True)
            x_val, ye_val, yc_val, t_val, e_val = (x_val.to(device), ye_val.to(device), yc_val.to(device),
                                                   t_val.to(device), e_val.to(device))

        best_loss = float('inf')
        best_ep = -1

        # training and evaluation
        prefix = f'Training w Early Stop on {device}' if early_stop else f'Training on {device} w/o Early Stop'
        pbar = trange(epochs, disable=not verbose, desc=prefix)
        for ep in pbar:
            # start training
            self.train()
            train_loss_ep = 0
            for xi, yei, yci, ti, ei in train_dataloader:
                xi, yei, yci, ti, ei = xi.to(device), yei.to(device), yci.to(device), ti.to(device), ei.to(device)
                ye_pred1, ye_pred2, yc_pred1, yc_pred2, epsilon, gamma, kappa = self(xi)

                # likelihood loss for event and censoring prediction
                loss = 0.25 * (self.loss(ye_pred1, yei) + self.loss(ye_pred2, yei)
                        + self.loss(yc_pred1, yci) + self.loss(yc_pred2, yci))
                loss += self.orthog_reg(self) if self.alpha > 0 else 0
                # loss += self.orthog_reg([epsilon, gamma, kappa]) if self.alpha > 0 else 0
                loss += self.balance_reg(epsilon, ti, 1 - ei, event=False) if self.beta > 0 else 0
                loss += self.balance_reg(gamma, ti, ei, event=True) if self.beta > 0 else 0

                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss_ep += loss.detach().item()

            scheduler.step()
            train_loss_ep /= len(train_dataloader)
            # evaluation
            self.eval()
            with torch.no_grad():
                postfix = f"Train loss = {train_loss_ep:.4f};"

                if early_stop and not val_df.empty:
                    ye_val_pred1, ye_val_pred2, yc_val_pred1, yc_val_pred2, eps_val, gam_val, kap_val = self(x_val)
                    eval_ll = 0.25 * (self.loss(ye_val_pred1, ye_val) + self.loss(ye_val_pred2, ye_val)
                                 + self.loss(yc_val_pred1, yc_val) + self.loss(yc_val_pred2, yc_val))
                    eval_orth = self.orthog_reg(self) if self.alpha > 0 else 0
                    # eval_orth = self.orthog_reg([eps_val, gam_val, kap_val]) if self.alpha > 0 else 0
                    eval_blc = self.balance_reg(eps_val, t_val, 1- e_val, event=False) if self.beta > 0 else 0
                    eval_blc += self.balance_reg(gam_val, t_val, e_val, event=True) if self.beta > 0 else 0
                    postfix += f" Val nll = {eval_ll:.4f}, orth = {eval_orth:.4f}, blc = {eval_blc:.4f};"
                    eval_loss = eval_ll + eval_orth + eval_blc

                    if best_loss > eval_loss:
                        best_loss = eval_loss
                        best_ep = ep
                        torch.save({'model_state_dict': self.state_dict()}, fname + '.pth')
                    if (ep - best_ep) > patience:
                        postfix += f"\nEarly stop at epoch {ep}. Best epoch is {best_ep}. Start testing..."
                        pbar.set_postfix_str(postfix)
                        break
                pbar.set_postfix_str(postfix)
        self.load_state_dict(torch.load(fname + '.pth')['model_state_dict']) if early_stop and not val_df.empty else None
