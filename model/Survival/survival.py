import pandas as pd
import torch
from torch import nn as nn
from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike

from model.Survival.base import BaseMTLR, ContinuousSurvivalBase, DiscreteSurvivalBase
from model.utils import build_sequential_nn
from model.loss import LikelihoodMTLR, PartialLikelihood, CensoredPinballLoss
from utils.util_survival import extract_survival, baseline_hazard


class MTLR(DiscreteSurvivalBase):
    """MTLR model with regularization"""

    def __init__(
            self,
            n_features: int,
            time_bins: torch.Tensor,
            hidden_size: list,
            norm: bool,
            activation: str,
            dropout: float
    ):
        super(MTLR, self).__init__(n_features, hidden_size, norm, activation, dropout)
        output_size = len(time_bins)
        self.time_bins = time_bins
        self.output_size = output_size
        self.loss = LikelihoodMTLR(reduction='mean')

        self.model = self._build_model()

    def _build_model(self):
        if not self.hidden_size:
            layers = [BaseMTLR(self.in_features, self.output_size)]
        else:
            layers = build_sequential_nn(self.in_features, self.hidden_size, self.norm, self.activation, self.dropout)
            layers.append(BaseMTLR(self.hidden_size[-1], self.output_size))
        return nn.Sequential(*layers)

    def predict_survival(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            assert logits.dim() == 2, "The logits should have dimension with size (n_data, n_bins)"
            G = torch.tril(torch.ones(logits.shape[1], logits.shape[1])).to(logits.device)
            density = torch.softmax(logits, dim=1)
            return torch.matmul(density, G)


class CoxPH(DiscreteSurvivalBase):
    """CoxPH model with regularization"""
    def __init__(
            self,
            n_features: int,
            hidden_size: list,
            norm: bool,
            activation: str,
            dropout: float
    ):
        super(CoxPH, self).__init__(n_features, hidden_size, norm, activation, dropout)
        self.output_size = 1
        self.time_bins = None
        self.baseline_hazard = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.loss = PartialLikelihood(reduction='mean')

        self.model = self._build_model()

    def _build_model(self):
        if not self.hidden_size:
            # if hidden_size is empty, then the only layer is linear
            layers = [nn.Linear(self.in_features, self.output_size)]
        else:
            layers = build_sequential_nn(self.in_features, self.hidden_size, self.norm, self.activation, self.dropout)
            layers.append(nn.Linear(self.hidden_size[-1], self.output_size))
        return nn.Sequential(*layers)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, device: torch.device, optimizer: str, batch_size: int,
            epochs: int, lr: float, lr_min: float, weight_decay: float, early_stop: bool = True,
            patience: int = 50, fname: str = '', verbose: bool = True):
        super(CoxPH, self).fit(train_df, val_df, device, optimizer, batch_size, epochs, lr, lr_min, weight_decay,
                               early_stop, patience, fname, verbose)
        self.cal_baseline_survival(train_df)

    def predict_risk(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def predict_survival(self, x):
        self.eval()
        with torch.no_grad():
            risks = self.predict_risk(x)
            n_data = len(risks)
            risk_score = torch.exp(risks)
            risk_score = risk_score.reshape(-1)
            survival_curves = torch.empty((n_data, self.baseline_survival.shape[0]), dtype=torch.double).to(
                risks.device)
            for i in range(n_data):
                survival_curve = torch.pow(self.baseline_survival, risk_score[i])
                survival_curves[i] = survival_curve
            return survival_curves

    def cal_baseline_survival(self, dataset):
        x, _, t, e = extract_survival(dataset)
        device = next(self.parameters()).device
        x, t, e = x.to(device), t.to(device), e.to(device)
        with torch.no_grad():
            outputs = self.forward(x)
        self.time_bins, self.baseline_hazard, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)


class CenQuanRegNN(DiscreteSurvivalBase):
    """Censored Quantile Regression Neural Network"""
    def __init__(
            self,
            n_features: int,
            hidden_size: list,
            n_quantiles: int,
            norm: bool,
            activation: str,
            dropout: float,
            t_max: float):
        super(CenQuanRegNN, self).__init__(n_features, hidden_size, norm, activation, dropout)
        if n_quantiles is None:
            self.output_size = 9
        else:
            assert isinstance(n_quantiles, int) and n_quantiles > 0, "n_quantiles must be a positive integer"
            self.output_size = n_quantiles
        self.quan_levels = torch.linspace(1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1),
                                          n_quantiles, dtype=torch.float64)
        self.time_bins = None
        self.loss = CensoredPinballLoss(self.quan_levels, use_cross_loss=True, reduction='mean')
        self.loss.t_max = t_max

        self.model = self._build_model()

    def _build_model(self):
        if not self.hidden_size:
            layers = [nn.Linear(self.in_features, self.output_size)]
        else:
            layers = build_sequential_nn(self.in_features, self.hidden_size, self.norm, self.activation, self.dropout)
            layers.append(nn.Linear(self.hidden_size[-1], self.output_size))
        return nn.Sequential(*layers)

    def predict_time(self, x, pred_type='median'):
        if pred_type == 'mean':
            raise NotImplementedError("Mean prediction is not implemented for CenQuanRegNN.")
        elif pred_type == 'median':
            if 0.5 in self.quan_levels:
                return self.forward(x)[:, self.quan_levels == 0.5]
        elif pred_type == 'rmst':
            raise NotImplementedError("RMST prediction is not implemented for CenQuanRegNN.")

    def predict_quantiles(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def predict_survival(self, x):
        quantiles = self.predict_quantiles(x)
        return quantiles


class WeibullAFT(ContinuousSurvivalBase):
    """Weibull Accelerated Failure Time model, also known as Weibull Proportional Hazard model, or Weibull regression"""
    def __init__(
            self,
            n_features: int,
            hidden_size: list,
            norm: bool,
            activation: str,
            dropout: float,
            time_grids: NumericArrayLike=None
    ):
        super(WeibullAFT, self).__init__(n_features, hidden_size, norm, activation, dropout, time_grids)
        # trainable parameters: shape, scale, and network weights
        self._alpha = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self._lambda = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    def _predict_survival(self, x, times):
        return torch.exp(- self._predict_cum_hazard(x, times))

    def _predict_pdf(self, x, times):
        return self._predict_survival(x, times) * self._predict_hazard(x, times)

    def _predict_hazard(self, x, times):
        risks = torch.exp(self.forward(x))
        ndim = times.dim()
        if ndim == 1:
            risks = risks.squeeze()
        elif ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected dimension of times: {ndim}.")

        return self._alpha * self._lambda * torch.pow(times, self._alpha - 1) * risks

    def _predict_cum_hazard(self, x, times):
        risks = torch.exp(self.forward(x))
        ndim = times.dim()
        if ndim == 1:
            risks = risks.squeeze()
        elif ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected dimension of times: {ndim}.")
        return self._lambda * torch.pow(times, self._alpha) * risks

    def _predict_quantiles(self, x, quantiles):
        assert quantiles.dim() == 1, "Quantiles must be a 1D tensor"
        risks = torch.exp(self.forward(x))
        return torch.pow(- torch.log(1 - quantiles) / (self._lambda * risks), 1 / self._alpha)

    def reset_parameters(self):
        self._alpha.data.fill_(1.0)
        self._lambda.data.fill_(1.0)
        super(WeibullAFT, self).reset_parameters()


class LogLogisticAFT(ContinuousSurvivalBase):
    """Log Logistic Accelerated Failure Time model, also known as Log Logistic regression"""
    def __init__(
            self,
            n_features: int,
            hidden_size: list,
            norm: bool,
            activation: str,
            dropout: float,
            time_grids: NumericArrayLike=None
    ):
        super(LogLogisticAFT, self).__init__(n_features, hidden_size, norm, activation, dropout, time_grids)
        # trainable parameters
        self._alpha = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self._lambda = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    def _predict_survival(self, x, times):
        risks = torch.exp(self.forward(x))
        ndim = times.dim()
        if ndim == 1:
            risks = risks.squeeze()
        elif ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected dimension of times: {ndim}.")
        return 1 / (1 + self._lambda * torch.pow(times, self._alpha) * risks)

    def _predict_pdf(self, x, times):
        return self._predict_survival(x, times) * self._predict_hazard(x, times)

    def _predict_hazard(self, x, times):
        risks = torch.exp(self.forward(x))
        ndim = times.dim()
        if ndim == 1:
            risks = risks.squeeze()
        elif ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected dimension of times: {ndim}.")
        return (self._lambda * self._alpha * torch.pow(times, self._alpha - 1) * risks /
                (1 + self._lambda * torch.pow(times, self._alpha) * risks))

    def _predict_cum_hazard(self, x, times):
        return - torch.log(1 - self._predict_survival(x, times))

    def _predict_quantiles(self, x, quantiles):
        assert quantiles.dim() == 1, "Quantiles must be a 1D tensor"
        risks = torch.exp(self.forward(x))
        return torch.pow(quantiles / ((1 - quantiles) * self._lambda * risks), 1 / self._alpha)

    def reset_parameters(self):
        self._alpha.data.fill_(1.0)
        self._lambda.data.fill_(1.0)
        super(LogLogisticAFT, self).reset_parameters()
