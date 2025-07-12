import pandas as pd
import torch
from torch import nn as nn
from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike

from model.TwoBranch.base import BaseMTLR2B, ContinuousSurvivalBase2B, DiscreteSurvivalBase2B
from model.utils import build_sequential_nn
from model.loss import LikelihoodMTLR, PartialLikelihood, CensoredPinballLoss
from utils.util_survival import extract_survival, baseline_hazard


class MTLR2B(DiscreteSurvivalBase2B):
    """Multi-task Logistic Regression model with two branches"""
    def __init__(
            self,
            n_features: int,
            time_bins_event: torch.Tensor,
            time_bins_censor: torch.Tensor,
            hidden_size: list,
            norm: bool,
            activation: str,
            dropout: float
    ):
        super(MTLR2B, self).__init__(n_features, hidden_size, norm, activation, dropout)
        self.output_size_event = len(time_bins_event)
        self.output_size_censor = len(time_bins_censor)
        self.output_size = self.output_size_event + self.output_size_censor
        self.time_bins_e = time_bins_event
        self.time_bins_c = time_bins_censor
        self.time_bins = self.time_bins_e
        self.loss = LikelihoodMTLR(reduction='mean')
        self.model = self._build_model()

    def _build_model(self):
        if not self.hidden_size:
            layers = [BaseMTLR2B(self.in_features, self.output_size_event, self.output_size_censor)]
        else:
            layers = build_sequential_nn(self.in_features, self.hidden_size, self.norm, self.activation, self.dropout)
            layers.append(BaseMTLR2B(self.hidden_size[-1], self.output_size_event, self.output_size_censor))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        return output[:, :self.output_size_event + 1], output[:, self.output_size_event + 1:]   # +1 for the extra time bin [max_time, inf)

    def predict_survival(self, x, target='e'):
        self.eval()
        with torch.no_grad():
            if target == 'e':
                logits, _ = self.forward(x)
                msg = "n_event_bins"
            elif target == 'c':
                _, logits = self.forward(x)
                msg = "n_censor_bins"
            else:
                raise ValueError(f"Invalid target: {target}")
            assert logits.dim() == 2, f"The logits should have dimension with size (n_data, {msg})"
            G = torch.tril(torch.ones(logits.shape[1], logits.shape[1])).to(logits.device)
            density = torch.softmax(logits, dim=1)
            return torch.matmul(density, G)


class CoxPH2B(DiscreteSurvivalBase2B):
    """Cox Proportional Hazard model with two branches"""
    def __init__(self, n_features: int, hidden_size: list, norm: bool, activation: str, dropout: float):
        super(CoxPH2B, self).__init__(n_features, hidden_size, norm, activation, dropout)
        self.output_size_event = 1
        self.output_size_censor = 1
        self.output_size = 2
        self.time_bins_e = None
        self.time_bins_c = None
        self.time_bins = self.time_bins_e
        self.baseline_hazard_e = None
        self.baseline_hazard_c = None
        self.cum_baseline_hazard_e = None
        self.cum_baseline_hazard_c = None
        self.baseline_survival_e = None
        self.baseline_survival_c = None
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
        super(CoxPH2B, self).fit(train_df, val_df, device, optimizer, batch_size, epochs, lr, lr_min, weight_decay,
                                 early_stop, patience, fname, verbose)
        self.cal_baseline_survival(train_df)

    def predict_risk(self, x, target='e'):
        self.eval()
        with torch.no_grad():
            if target == 'e':
                risks, _ = self.forward(x)
            elif target == 'c':
                _, risks = self.forward(x)
            else:
                raise ValueError(f"Invalid target: {target}")

            return risks

    def predict_survival(self, x, target='e'):
        self.eval()
        with torch.no_grad():
            risks = self.predict_risk(x, target=target)
            n_data = len(risks)
            risk_score = torch.exp(risks)
            risk_score = risk_score.reshape(-1)
            bl = self.baseline_survival_e if target == 'e' else self.baseline_survival_c
            survival_curves = torch.empty((n_data, bl.shape[0]), dtype=torch.double).to(risks.device)
            for i in range(n_data):
                survival_curve = torch.pow(bl, risk_score[i])
                survival_curves[i] = survival_curve
            return survival_curves

    def cal_baseline_survival(self, dataset):
        x, _, t, e = extract_survival(dataset)
        device = next(self.parameters()).device
        x, t, e = x.to(device), t.to(device), e.to(device)
        with torch.no_grad():
            outputs_e, outputs_c = self.forward(x)
        self.time_bins_e, self.baseline_hazard_e, self.cum_baseline_hazard_e, self.baseline_survival_e = baseline_hazard(
            outputs_e, t, e)
        self.time_bins = self.time_bins_e
        self.time_bins_c, self.baseline_hazard_c, self.cum_baseline_hazard_c, self.baseline_survival_c = baseline_hazard(
            outputs_c, t, 1 - e)


class CenQuanRegNN2B(DiscreteSurvivalBase2B):
    """Censored Quantile Regression Neural Network with two branches"""
    def __init__(
            self,
            n_features: int,
            hidden_size: list,
            n_quantiles: int,
            norm: bool,
            activation: str,
            dropout: float,
            t_max: float
    ):
        super(CenQuanRegNN2B, self).__init__(n_features, hidden_size, norm, activation, dropout)
        self.output_size_event = n_quantiles
        self.output_size_censor = n_quantiles
        self.output_size = n_quantiles * 2
        # quantile levels are the same for both event and censoring branches
        self.quan_levels = torch.linspace(1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1),
                                          n_quantiles, dtype=torch.float64)
        self.time_bins_e = None
        self.time_bins_c = None
        self.time_bins = self.time_bins_e
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

    def predict_quantiles(self, x, target='e'):
        self.eval()
        with torch.no_grad():
            if target == 'e':
                qe, _ = self.forward(x)
                return qe
            elif target == 'c':
                _, qc = self.forward(x)
                return qc
            else:
                raise ValueError(f"Invalid target: {target}")

    def predict_survival(self, x, target='e'):
        quantiles = self.predict_quantiles(x, target=target)
        return quantiles


class WeibullAFT2B(ContinuousSurvivalBase2B):
    """Weibull Accelerated Failure Time model with two branches"""
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
        super(WeibullAFT2B, self).__init__(n_features, hidden_size, norm, activation, dropout, time_grids_e, time_grids_c)
        # training parameters: 2 sets of (shape, scale) parameters, and network weights
        # the first set is for event, and the second set is for censoring
        self._alpha = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))
        self._lambda = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))

    def _predict_survival(self, x, times, target='both'):
        cum_hazard_e, cum_hazard_c = self._predict_cum_hazard(x, times, 'both')
        if target == 'both':
            return torch.exp(-cum_hazard_e), torch.exp(-cum_hazard_c)
        elif target == 'e':
            return torch.exp(-cum_hazard_e)
        elif target == 'c':
            return torch.exp(-cum_hazard_c)
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_pdf(self, x, times, target='both'):
        hazard_e, hazard_c = self._predict_hazard(x, times, 'both')
        survival_e, survival_c = self._predict_survival(x, times, 'both')
        if target == 'both':
            return hazard_e * survival_e, hazard_c * survival_c
        elif target == 'e':
            return hazard_e * survival_e
        elif target == 'c':
            return hazard_c * survival_c
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_hazard(self, x, times, target='both'):
        output_e, output_c = self.forward(x)
        risks_e, risks_c = torch.exp(output_e), torch.exp(output_c)
        ndim = times.dim()
        if ndim == 1:
            risks_e = risks_e.squeeze()
            risks_c = risks_c.squeeze()
        elif ndim == 2:
            pass
        else:
            raise ValueError(f"Invalid dimension: {ndim}")

        if target == 'both':
            hazard_e = self._alpha[0] * self._lambda[0] * torch.pow(times, self._alpha[0] - 1) * risks_e
            hazard_c = self._alpha[1] * self._lambda[1] * torch.pow(times, self._alpha[1] - 1) * risks_c
            return hazard_e, hazard_c
        elif target == 'e':
            return self._alpha[0] * self._lambda[0] * torch.pow(times, self._alpha[0] - 1) * risks_e
        elif target == 'c':
            return self._alpha[1] * self._lambda[1] * torch.pow(times, self._alpha[1] - 1) * risks_c
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_cum_hazard(self, x, times, target='both'):
        output_e, output_c = self.forward(x)
        risks_e, risks_c = torch.exp(output_e), torch.exp(output_c)
        ndim = times.dim()
        if ndim == 1:
            risks_e = risks_e.squeeze()
            risks_c = risks_c.squeeze()
        elif ndim == 2:
            pass
        else:
            raise ValueError(f"Invalid dimension: {ndim}")

        if target == 'both':
            cum_hazard_e = self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e
            cum_hazard_c = self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c
            return cum_hazard_e, cum_hazard_c
        elif target == 'e':
            return self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e
        elif target == 'c':
            return self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_quantiles(self, x, quantiles, target='both'):
        assert quantiles.dim() == 1, "Quantiles should be a 1D tensor."
        output_e, output_c = self.forward(x)
        risks_e, risks_c = torch.exp(output_e), torch.exp(output_c)
        if target == 'both':
            q_e = torch.pow(-torch.log(1 - quantiles) / (self._lambda[0] * risks_e), 1 / self._alpha[0])
            q_c = torch.pow(-torch.log(1 - quantiles) / (self._lambda[1] * risks_c), 1 / self._alpha[1])
            return q_e, q_c
        elif target == 'e':
            return torch.pow(-torch.log(1 - quantiles) / (self._lambda[0] * risks_e), 1 / self._alpha[0])
        elif target == 'c':
            return torch.pow(-torch.log(1 - quantiles) / (self._lambda[1] * risks_c), 1 / self._alpha[1])
        else:
            raise ValueError(f"Invalid target: {target}")

    def reset_parameters(self):
        self._alpha.data.fill_(1.0)
        self._lambda.data.fill_(1.0)
        super(WeibullAFT2B, self).reset_parameters()


class LogLogisticAFT2B(ContinuousSurvivalBase2B):
    """Log-Logistic Accelerated Failure Time model with two branches"""
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
        super(LogLogisticAFT2B, self).__init__(n_features, hidden_size, norm, activation, dropout, time_grids_e, time_grids_c)
        # training parameters: 2 sets of (shape, scale) parameters, and network weights
        # the first set is for event, and the second set is for censoring
        self._alpha = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))
        self._lambda = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))

    def _predict_survival(self, x, times, target='both'):
        output_e, output_c = self.forward(x)
        risks_e, risks_c = torch.exp(output_e), torch.exp(output_c)
        ndim = times.dim()
        if ndim == 1:
            risks_e = risks_e.squeeze()
            risks_c = risks_c.squeeze()
        elif ndim == 2:
            pass
        else:
            raise ValueError(f"Invalid dimension: {ndim}")

        if target == 'both':
            survival_e = 1 / (1 + self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e)
            survival_c = 1 / (1 + self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c)
            return survival_e, survival_c
        elif target == 'e':
            return 1 / (1 + self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e)
        elif target == 'c':
            return 1 / (1 + self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c)
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_pdf(self, x, times, target='both'):
        survival_e, survival_c = self._predict_survival(x, times, 'both')
        hazard_e, hazard_c = self._predict_hazard(x, times, 'both')
        if target == 'both':
            return hazard_e * survival_e, hazard_c * survival_c
        elif target == 'e':
            return hazard_e * survival_e
        elif target == 'c':
            return hazard_c * survival_c
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_hazard(self, x, times, target='both'):
        output_e, output_c = self.forward(x)
        risks_e, risks_c = torch.exp(output_e), torch.exp(output_c)
        ndim = times.dim()
        if ndim == 1:
            risks_e = risks_e.squeeze()
            risks_c = risks_c.squeeze()
        elif ndim == 2:
            pass
        else:
            raise ValueError(f"Invalid dimension: {ndim}")

        if target == 'both':
            hazard_e = (self._lambda[0] * self._alpha[0] * torch.pow(times, self._alpha[0] - 1) * risks_e /
                        (1 + self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e))
            hazard_c = (self._lambda[1] * self._alpha[1] * torch.pow(times, self._alpha[1] - 1) * risks_c /
                        (1 + self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c))
            return hazard_e, hazard_c
        elif target == 'e':
            return (self._lambda[0] * self._alpha[0] * torch.pow(times, self._alpha[0] - 1) * risks_e /
                    (1 + self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e))
        elif target == 'c':
            return (self._lambda[1] * self._alpha[1] * torch.pow(times, self._alpha[1] - 1) * risks_c /
                    (1 + self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c))
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_cum_hazard(self, x, times, target='both'):
        survival_e, survival_c = self._predict_survival(x, times, 'both')
        if target == 'both':
            return -torch.log(1 - survival_e), -torch.log(1 - survival_c)
        elif target == 'e':
            return -torch.log(1 - survival_e)
        elif target == 'c':
            return -torch.log(1 - survival_c)
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_quantiles(self, x, quantiles, target='both'):
        assert quantiles.dim() == 1, "Quantiles should be a 1D tensor."
        output_e, output_c = self.forward(x)
        risks_e, risks_c = torch.exp(output_e), torch.exp(output_c)
        if target == 'both':
            q_e = torch.pow(quantiles / ((1 - quantiles) * self._lambda[0] * risks_e), 1 / self._alpha[0])
            q_c = torch.pow(quantiles / ((1 - quantiles) * self._lambda[1] * risks_c), 1 / self._alpha[1])
            return q_e, q_c
        elif target == 'e':
            return torch.pow(quantiles / ((1 - quantiles) * self._lambda[0] * risks_e), 1 / self._alpha[0])
        elif target == 'c':
            return torch.pow(quantiles / ((1 - quantiles) * self._lambda[1] * risks_c), 1 / self._alpha[1])
        else:
            raise ValueError(f"Invalid target: {target}")

    def reset_parameters(self):
        self._alpha.data.fill_(1.0)
        self._lambda.data.fill_(1.0)
        super(LogLogisticAFT2B, self).reset_parameters()
