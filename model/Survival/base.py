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


class BaseMTLR(nn.Module):
    """(Linear) Multi-task logistic regression model.

    Code is adapted from https://github.com/mkazmier/torchmtlr
    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time
    interval.

    Note that a slightly more efficient reformulation is used here, first
    proposed in [2].

    References
    ----------
    [1] C.-N. Yu et al., Learning patient-specific cancer survival
    distributions as a sequence of dependent regressors, in Advances in neural
    information processing systems 24, 2011, pp. 1845–1853.
    [2] P. Jin, Using Survival Prediction Techniques to Learn
    Consumer-Specific Reservation Price Distributions, Master's thesis,
    University of Alberta, Edmonton, AB, 2015.
    """

    def __init__(self, in_features: int, num_time_bins: int):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        """
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.num_time_bins - 1, self.in_features))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))
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


class SurvivalBase(nn.Module):
    """Base class for standard survival models."""
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

    @abstractmethod
    def predict_survival(self, x):
        pass

    def predict_cdf(self, x):
        return 1 - self.predict_survival(x)

    @abstractmethod
    def predict_time(self, x, pred_type='mean'):
        pass

    def forward(self, x):
        return self.model(x)

    def reset_parameters(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class ContinuousSurvivalBase(SurvivalBase):
    """Base class for survival models which generates continuous survival distributions."""

    def __init__(
            self,
            n_features: int,
            hidden_size: list,
            norm: bool,
            activation: str,
            dropout: float,
            time_grids: NumericArrayLike=None
    ):
        super().__init__(n_features, hidden_size, norm, activation, dropout)
        self.output_size = 1
        # time_grids is the time points to generate the survival function
        if time_grids is None:
            warnings.warn("'time_grids' is not provided, the time grids will be generated using the time points "
                  "in the training data.")
        self.t_grids = time_grids
        self.time_bins = None
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

    def predict_pdf(self, x):
        self.eval()
        with torch.no_grad():
            t_grids = self.t_grids.repeat(x.shape[0], 1)
            return self._predict_pdf(x, t_grids)

    def predict_survival(self, x):
        self.eval()
        with torch.no_grad():
            t_grids = self.t_grids.repeat(x.shape[0], 1)
            return self._predict_survival(x, t_grids)

    def predict_hazard(self, x):
        self.eval()
        with torch.no_grad():
            t_grids = self.t_grids.repeat(x.shape[0], 1)
            return self._predict_hazard(x, t_grids)

    def predict_cum_hazard(self, x):
        self.eval()
        with torch.no_grad():
            t_grids = self.t_grids.repeat(x.shape[0], 1)
            return self._predict_cum_hazard(x, t_grids)

    def predict_quantiles(self, x, quantiles):
        self.eval()
        with torch.no_grad():
            return self._predict_quantiles(x, quantiles)

    def predict_time(self, x, pred_type='mean'):
        self.eval()
        with torch.no_grad():
            if pred_type == 'mean':
                raise NotImplementedError
            elif pred_type == 'median':
                return self.predict_quantiles(x, 0.5)
            elif pred_type == 'rmst':
                # Do we need to calculate the RMST for continuous survival models?
                raise NotImplementedError
            else:
                raise ValueError("pred_type should be either 'mean', 'median' or 'rmst'")

    @abstractmethod
    def _predict_pdf(self, x, times):
        pass

    @abstractmethod
    def _predict_survival(self, x, times):
        pass

    @abstractmethod
    def _predict_hazard(self, x, times):
        pass

    @abstractmethod
    def _predict_cum_hazard(self, x, times):
        pass

    @abstractmethod
    def _predict_quantiles(self, x, quantiles):
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

        self.t_grids = torch.tensor(train_df['time'], dtype=torch.float64).to(device).unique()
        self.t_grids = torch.cat([torch.tensor([0.], dtype=torch.float64).to(device), self.t_grids], dim=0)

        optimizer = getattr(torch.optim, optimizer)
        optim = optimizer((param for param in self.parameters() if param.requires_grad), lr=lr,
                          weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=lr_min)
        x_train, y_train, _, _ = extract_survival(train_df, self.time_bins)
        train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

        if not val_df.empty:
            x_val, y_val, t_val, _ = extract_survival(val_df, self.time_bins)
            x_val, y_val, t_val = x_val.to(device), y_val.to(device), t_val.to(device)

        best_loss = float('inf')
        best_ep = -1

        # training and evaluation
        prefix = f'Training w Early Stop on {device}' if early_stop else f'Training on {device} w/o Early Stop'
        pbar = trange(epochs, disable=not verbose, desc=prefix)
        for ep in pbar:
            # start training
            self.train()
            train_loss_ep = 0
            for xi, yi in train_dataloader:
                xi, yi = xi.to(device), yi.to(device)

                ti = yi[:, 0]
                survival = self._predict_survival(xi, ti)
                pdf = self._predict_pdf(xi, ti)
                loss = self.loss(survival, pdf, yi)

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
                    survival_val_pred = self._predict_survival(x_val, t_val)
                    pdf_val_pred = self._predict_pdf(x_val, t_val)
                    eval_loss = self.loss(survival_val_pred, pdf_val_pred, y_val)
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


class DiscreteSurvivalBase(SurvivalBase):
    """Base class for survival models which generates discrete survival distributions."""
    def __init__(
            self,
            n_features: int,
            hidden_size: list,
            norm: bool,
            activation: str,
            dropout: float,
    ):
        super().__init__(n_features, hidden_size, norm, activation, dropout)
        self._output_size = None

    @property
    def output_size(self):
        return self._output_size

    @output_size.setter
    def output_size(self, value):
        self._output_size = value

    def predict_time(self, x, pred_type='mean'):
        survival = self.predict_survival(x)
        if pred_type == 'rmst':
            # restricted mean survival time (rmst)
            # integral the survival function
            return torch.trapz(survival, torch.cat([torch.tensor([0]), self.time_bins], dim=0).to(survival.device), dim=1)
        elif pred_type == 'median':
            # fit the survival function using spline, then find the median
            raise NotImplementedError
        elif pred_type == 'mean':
            # mean survival time = rmst + residual
            # rmst is the area under the survival function
            # residual is the area under the extrapolated survival function
            # extrapolate survival function by extending the last time bin to where the survival function is 0
            rmst = torch.trapz(survival, torch.cat([torch.tensor([0]), self.time_bins], dim=0).to(survival.device), dim=1)
            # calculate the extended time when the linear extension of survival function reaches 0
            extend_time = (self.time_bins[-1] - self.time_bins[0]) / (survival[:, 0] - survival[:, -1])
            residual = 0.5 * (extend_time - self.time_bins[-1]) * survival[:, -1]
            return rmst + residual
        else:
            raise ValueError("pred_type should be either 'mean', 'median' or 'rmst'")

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
        x_train, y_train, _, _ = extract_survival(train_df, self.time_bins)
        train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

        if not val_df.empty:
            x_val, y_val, _, _ = extract_survival(val_df, self.time_bins)
            x_val, y_val = x_val.to(device), y_val.to(device)

        best_loss = float('inf')
        best_ep = -1

        # training and evaluation
        prefix = f'Training w Early Stop on {device}' if early_stop else f'Training on {device} w/o Early Stop'
        pbar = trange(epochs, disable=not verbose, desc=prefix)
        for ep in pbar:
            # start training
            self.train()
            train_loss_ep = 0
            for xi, yi in train_dataloader:
                xi, yi = xi.to(device), yi.to(device)
                y_pred = self(xi)

                loss = self.loss(y_pred, yi)

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
                    y_val_pred = self(x_val)
                    eval_loss = self.loss(y_val_pred, y_val)
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
