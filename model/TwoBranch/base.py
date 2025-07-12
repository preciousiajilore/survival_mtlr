from abc import abstractmethod
import warnings
import pandas as pd
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike

from model.loss import Likelihood
from model.utils import build_sequential_nn
from utils.util_survival import extract_survival


class BaseMTLR2B(nn.Module):
    """
    (Linear) Multi-task logistic regression model with two branches (event and censoring).

    It first uses a NN to generate the hidden representation. The hidden representation is
    shared between the two branches. Then, it uses two (independent but parallel) linear MTLR
    layers to generate the logits/outputs.

    The two independent and parallel MTLR layers are implemented using one instances of BaseMTLR2B.
    The output of the BaseMTLR2B is the concatenation of the two MTLR outputs. The first part of the
    output is the event logits, and the second part is the censoring logits.
    """
    def __init__(self, in_features: int, num_time_bins_e: int, num_time_bins_c: int):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins_e
            The number of bins to divide the event time axis into.
        num_time_bins_c
            The number of bins to divide the censoring time axis
        """
        super().__init__()
        if num_time_bins_e < 1 or num_time_bins_c < 1:
            raise ValueError("The number of time bins must be both greater than 1")
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.in_features = in_features
        self.num_time_bins = (num_time_bins_e + 1) + (num_time_bins_c + 1)  # + extra time bin [max_time, inf)

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.num_time_bins - 2, self.in_features))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 2))

        # `G` is the coding matrix inspired from [2]_ used for fast summation.
        # It is a block diagonal matrix with the first block being the coding matrix for event time
        # and the second block being the coding matrix for censoring time.
        self.register_buffer(
            "G",
            torch.block_diag(
                torch.tril(torch.ones(num_time_bins_e, num_time_bins_e + 1, requires_grad=True)),
                torch.tril(torch.ones(num_time_bins_c, num_time_bins_c + 1, requires_grad=True)),
            )
        )
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on a batch of examples.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.

        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins - 1)
            The predicted time logits.
        """
        out = F.linear(x, self.mtlr_weight, self.mtlr_bias)
        return torch.matmul(out, self.G)

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, num_time_bins={self.num_time_bins})"


class SurvivalBase2B(nn.Module):
    """
    Base class for survival models which generates both event and censoring distributions (aka two branches, 2B).

    It first uses a NN to generate the hidden representation. The hidden representation is
    shared between the two branches. Then, it uses two (independent but parallel) linear
    layers to generate the logits/outputs.
    """
    def __init__(
            self,
            n_features: int,
            hidden_size: list,
            norm: bool,
            activation: str,
            dropout: float,
    ):
        super().__init__()
        self.in_features = n_features
        self.dropout = dropout
        self.norm = norm
        self.hidden_size = hidden_size
        self.activation = activation
        self.loss = None
        self.model = None
        # Add self.time_bins for discrete survival models
        self._output_size_event = None
        self._output_size_censor = None
        self._output_size = None

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
    def predict_survival(self, x, target='e'):
        pass

    def predict_cdf(self, x, target='e'):
        return 1 - self.predict_survival(x, target='e')

    @abstractmethod
    def predict_time(self, x, pred_type='mean'):
        pass

    def forward(self, x):
        output = self.model(x)
        return output[:, :self.output_size_event], output[:, self.output_size_event:]

    def reset_parameters(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class ContinuousSurvivalBase2B(SurvivalBase2B):
    """
    Base class for continuous survival models which generates both event and censoring distributions (aka two branches, 2B).
    """

    def __init__(
            self,
            n_features: int,
            hidden_size: list,
            norm: bool,
            activation: str,
            dropout: float,
            time_grids_e: NumericArrayLike=None,
            time_grids_c: NumericArrayLike=None
    ):
        super().__init__(n_features, hidden_size, norm, activation, dropout)
        self.output_size_event = 1
        self.output_size_censor = 1
        self.output_size = self.output_size_event + self.output_size_censor
        if time_grids_e is None or time_grids_c is None:
            warnings.warn("'time_grids' is not provided, the time grids will be generated using the time points"
                          "in the training data.")
        self.t_grids_e = time_grids_e
        self.t_grids_c = time_grids_c
        self.t_grids = self.t_grids_e
        self.time_bins_e = None
        self.time_bins_c = None
        self.time_bins = self.time_bins_e
        self.loss = Likelihood(reduction="mean")
        self.model = self._build_model()

    def _build_model(self):
        if not self.hidden_size:
            # if hidden_size is empty, then the only layer is linear
            layers = [nn.Linear(self.in_features, self.output_size)]
        else:
            layers = build_sequential_nn(self.in_features, self.hidden_size, self.norm, self.activation, self.dropout)
            layers.append(nn.Linear(self.hidden_size[-1], self.output_size))
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
    def _predict_pdf(self, x, times, target='both'):
        pass

    @abstractmethod
    def _predict_survival(self, x, times, target='both'):
        pass

    @abstractmethod
    def _predict_hazard(self, x, times, target='both'):
        pass

    @abstractmethod
    def _predict_cum_hazard(self, x, times, target='both'):
        pass

    @abstractmethod
    def _predict_quantiles(self, x, quantiles, target='both'):
        pass

    def fit(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            device: torch.device,
            optimizer: str,
            batch_size: int,
            epochs: int,
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

        if self.t_grids_e is None or self.t_grids_c is None:
            self.t_grids = torch.tensor(train_df['time'], dtype=torch.float64).to(device).unique()
            self.t_grids = torch.cat([torch.tensor([0.], dtype=torch.float64).to(device), self.t_grids], dim=0)
            self.t_grids_e = self.t_grids
            self.t_grids_c = self.t_grids

        optimizer = getattr(torch.optim, optimizer)
        optim = optimizer((param for param in self.parameters() if param.requires_grad), lr=lr,
                          weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=lr_min)
        x_train, ye_train, yc_train, _, _ = extract_survival(train_df, self.time_bins_e, self.time_bins_c,
                                                             include_censor_label=True)
        train_dataloader = DataLoader(TensorDataset(x_train, ye_train, yc_train),
                                      batch_size=batch_size, shuffle=True)

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
                survival_e, survival_c = self._predict_survival(xi, ti, 'both')
                pdf_e, pdf_c = self._predict_pdf(xi, ti, 'both')
                loss = 0.5 * (self.loss(survival_e, pdf_e, yei) + self.loss(survival_c, pdf_c, yci))

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
                    survival_val_pred_e, survival_val_pred_c = self._predict_survival(x_val, t_val, 'both')
                    pdf_val_pred_e, pdf_val_pred_c = self._predict_pdf(x_val, t_val, 'both')
                    eval_loss = 0.5 * (self.loss(survival_val_pred_e, pdf_val_pred_e, ye_val) +
                                       self.loss(survival_val_pred_c, pdf_val_pred_c, yc_val))
                    # eval_loss = self.loss(survival_val_pred_e, pdf_val_pred_e, ye_val)
                    postfix += f" Val loss = {eval_loss:.4f};"

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


class DiscreteSurvivalBase2B(SurvivalBase2B):
    """
    Base class for discrete survival models which generates both event and censoring distributions (aka two branches, 2B).

    It first uses a NN to generate the hidden representation. The hidden representation is
    shared between the two branches. Then, it uses two (independent but parallel) linear
    layers to generate the logits/outputs.
    """

    def predict_time(self, x, target='e', pred_type='mean'):
        survival = self.predict_survival(x, target)
        if pred_type == 'mean':
            raise NotImplementedError("Mean prediction is not implemented for discrete survival models.")
        elif pred_type == 'median':
            raise NotImplementedError("Median prediction is not implemented for discrete survival models.")
        elif pred_type == 'rmst':
            return

    def fit(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            device: torch.device,
            optimizer: str,
            batch_size: int,
            epochs: int,
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

        optimizer = getattr(torch.optim, optimizer)
        optim = optimizer((param for param in self.parameters() if param.requires_grad), lr=lr,
                          weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=lr_min)
        x_train, ye_train, yc_train, _, _ = extract_survival(train_df, self.time_bins_e, self.time_bins_c,
                                                             include_censor_label=True)
        train_dataloader = DataLoader(TensorDataset(x_train, ye_train, yc_train),
                                      batch_size=batch_size, shuffle=True)

        if not val_df.empty:
            x_val, ye_val, yc_val, _, _ = extract_survival(val_df, self.time_bins_e, self.time_bins_c,
                                                           include_censor_label=True)
            x_val, ye_val, yc_val = x_val.to(device), ye_val.to(device), yc_val.to(device)

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
                ye_pred, yc_pred = self(xi)

                loss = 0.5 * (self.loss(ye_pred, yei) + self.loss(yc_pred, yci))

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
                    ye_val_pred, yc_val_pred = self(x_val)
                    eval_loss = 0.5 * (self.loss(ye_val_pred, ye_val) + self.loss(yc_val_pred, yc_val))
                    postfix += f" Val loss = {eval_loss:.4f};"

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
