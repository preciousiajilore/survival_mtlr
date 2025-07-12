import pandas as pd
import torch
from torch.nn import functional as F
from torch import nn as nn
from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike

from model.LatDecReps.base import ContinuousSurvivalBaseLDR, DiscreteSurvivalBaseLDR
from model.loss import LikelihoodMTLR, PartialLikelihood, CensoredPinballLoss
from model.Survival.base import BaseMTLR
from model.utils import build_sequential_nn
from utils.util_survival import extract_survival, baseline_hazard


class MTLR_LDR(DiscreteSurvivalBaseLDR):
    """Multi-task Logistic Regression model via learning latent decomposed representations (LDR)."""
    def __init__(
            self,
            n_features: int,
            time_bins_event: torch.Tensor,
            time_bins_censor: torch.Tensor,
            rep_dims: list,
            event_dims: list,
            censor_dims: list,
            norm: bool,
            activation: str,
            dropout: float,
            ipm: str,
            alpha: float,
            beta: float
    ):
        super(MTLR_LDR, self).__init__(
            n_features=n_features,
            rep_dims=rep_dims,
            event_dims=event_dims,
            censor_dims=censor_dims,
            output_size_event=len(time_bins_event),
            output_size_censor=len(time_bins_censor),
            norm=norm,
            activation=activation,
            dropout=dropout,
            ipm=ipm,
            alpha=alpha,
            beta=beta
        )
        self.time_bins_e = time_bins_event
        self.time_bins_c = time_bins_censor
        self.time_bins = self.time_bins_e
        self.loss = LikelihoodMTLR(reduction='mean')

    def _build_dist_net(self, in_dim, hidden_dims, out_dim=1):
        if not hidden_dims:
            # if hidden_dims is empty, then the only layer is linear
            layers = [BaseMTLR(in_dim, out_dim)]
        else:
            layers = build_sequential_nn(in_dim, hidden_dims, self.norm, self.activation, self.dropout)
            layers.append(BaseMTLR(hidden_dims[-1], out_dim))
        return nn.Sequential(*layers)

    def predict_survival(self, x, target='e'):
        self.eval()
        with torch.no_grad():
            if target == 'e':
                logits = self.forward_e(x)
                msg = "n_event_bins"
            elif target == 'c':
                logits = self.forward_c(x)
                msg = "n_censor_bins"
            else:
                raise ValueError(f"Invalid target: {target}")
            assert logits.dim() == 2, f"The logits should have dimension with size (n_data, {msg})"
            G = torch.tril(torch.ones(logits.shape[1], logits.shape[1])).to(logits.device)
            density = torch.softmax(logits, dim=1)
            return torch.matmul(density, G)



class CoxPH_LDR(DiscreteSurvivalBaseLDR):
    """Cox Proportional Hazard model via learning latent decomposed representations (LDR)."""
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
            beta: float
    ):
        super(CoxPH_LDR, self).__init__(
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
        super(CoxPH_LDR, self).fit(train_df, val_df, device, batch_size, epochs, optimizer, lr, lr_min, weight_decay,
                                   early_stop, patience, fname, verbose)
        self.cal_baseline_survival(train_df)

    def predict_risk(self, x, target='e'):
        self.eval()
        with torch.no_grad():
            if target == 'e':
                risks = self.forward_e(x)
            elif target == 'c':
                risks = self.forward_c(x)
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
            _, outputs_e, _, outputs_c, _, _, _ = self.forward(x)
        self.time_bins_e, self.baseline_hazard_e, self.cum_baseline_hazard_e, self.baseline_survival_e = baseline_hazard(
            outputs_e, t, e)
        self.time_bins = self.time_bins_e
        self.time_bins_c, self.baseline_hazard_c, self.cum_baseline_hazard_c, self.baseline_survival_c = baseline_hazard(
            outputs_c, t, 1 - e)


class CenQuanRegNN_LDR(DiscreteSurvivalBaseLDR):
    """Censored Quantile Regression Neural Network model via learning latent decomposed representations (LDR)."""
    def __init__(
            self,
            n_features: int,
            rep_dims: list,
            event_dims: list,
            censor_dims: list,
            n_quantiles: int,
            norm: bool,
            activation: str,
            dropout: float,
            t_max: float,
            ipm: str,
            alpha: float,
            beta: float
    ):
        super(CenQuanRegNN_LDR, self).__init__(
            n_features=n_features,
            rep_dims=rep_dims,
            event_dims=event_dims,
            censor_dims=censor_dims,
            output_size_event=n_quantiles,
            output_size_censor=n_quantiles,
            norm=norm,
            activation=activation,
            dropout=dropout,
            ipm=ipm,
            alpha=alpha,
            beta=beta
        )
        # quantile levels are the same for both event and censoring branches
        self.quan_levels = torch.linspace(1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1),
                                          n_quantiles, dtype=torch.float64)
        self.time_bins_e = None
        self.time_bins_c = None
        self.time_bins = self.time_bins_e
        self.loss = CensoredPinballLoss(self.quan_levels, use_cross_loss=True, reduction='mean')
        self.loss.t_max = t_max

    def predict_quantiles(self, x, target='e'):
        self.eval()
        with torch.no_grad():
            if target == 'e':
                return self.forward_e(x)
            elif target == 'c':
                return self.forward_c(x)
            else:
                raise ValueError(f"Invalid target: {target}")

    def predict_survival(self, x, target='e'):
        quantiles = self.predict_quantiles(x, target=target)
        return quantiles


class WeibullAFT_LDR(ContinuousSurvivalBaseLDR):
    """Weibull Accelerated Failure Time model via learning latent decomposed representations (LDR)."""
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
        super(WeibullAFT_LDR, self).__init__(
            n_features=n_features,
            rep_dims=rep_dims,
            event_dims=event_dims,
            censor_dims=censor_dims,
            norm=norm,
            activation=activation,
            dropout=dropout,
            ipm=ipm,
            alpha=alpha,
            beta=beta,
            time_grids_e=time_grids_e,
            time_grids_c=time_grids_c
        )
        # training parameters: 2 sets of (shape, scale) parameters, and network weights
        # the first set is for event, and the second set is for censoring
        self._alpha = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))
        self._lambda = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))

    def _predict_survival(self, x, times, target='both'):
        if target == 'all':
            cum_hazard_e1, cum_hazard_e2, cum_hazard_c1, cum_hazard_c2, epsilon, gamma, kappa = self._predict_cum_hazard(
                x, times, 'all')
            return (torch.exp(-cum_hazard_e1), torch.exp(-cum_hazard_e2), torch.exp(-cum_hazard_c1),
                    torch.exp(-cum_hazard_c2), epsilon, gamma, kappa)
        elif target == 'both':
            cum_hazard_e2, cum_hazard_c2 = self._predict_cum_hazard(x, times, 'both')
            return torch.exp(-cum_hazard_e2), torch.exp(-cum_hazard_c2)
        elif target == 'e':
            cum_hazard_e2 = self._predict_cum_hazard(x, times, 'e')
            return torch.exp(-cum_hazard_e2)
        elif target == 'c':
            cum_hazard_c2 = self._predict_cum_hazard(x, times, 'c')
            return torch.exp(-cum_hazard_c2)
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_pdf(self, x, times, target='all'):
        if target == 'all':
            hazard_e1, hazard_e2, hazard_c1, hazard_c2, epsilon, gamma, kappa = self._predict_hazard(x, times, 'all')
            survival_e1, survival_e2, survival_c1, survival_c2, _, _, _ = self._predict_survival(x, times, 'all')
            return (hazard_e1 * survival_e1, hazard_e2 * survival_e2, hazard_c1 * survival_c1, hazard_c2 * survival_c2,
                    epsilon, gamma, kappa)
        elif target == 'both':
            hazard_e2, hazard_c2 = self._predict_hazard(x, times, 'both')
            survival_e2, survival_c2 = self._predict_survival(x, times, 'both')
            return hazard_e2 * survival_e2, hazard_c2 * survival_c2
        elif target == 'e':
            hazard_e2 = self._predict_hazard(x, times, 'e')
            survival_e2 = self._predict_survival(x, times, 'e')
            return hazard_e2 * survival_e2
        elif target == 'c':
            hazard_c2 = self._predict_hazard(x, times, 'c')
            survival_c2 = self._predict_survival(x, times, 'c')
            return hazard_c2 * survival_c2
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_hazard(self, x, times, target='both'):
        if target == 'all':
            risks_e1, risks_e2, risks_c1, risks_c2, epsilon, gamma, kappa = self._predict_risks(x, times, 'all')
            hazard_e1 = self._alpha[0] * self._lambda[0] * torch.pow(times, self._alpha[0] - 1) * risks_e1
            hazard_e2 = self._alpha[0] * self._lambda[0] * torch.pow(times, self._alpha[0] - 1) * risks_e2
            hazard_c1 = self._alpha[1] * self._lambda[1] * torch.pow(times, self._alpha[1] - 1) * risks_c1
            hazard_c2 = self._alpha[1] * self._lambda[1] * torch.pow(times, self._alpha[1] - 1) * risks_c2
            return hazard_e1, hazard_e2, hazard_c1, hazard_c2, epsilon, gamma, kappa
        elif target == 'both':
            risks_e2, risks_c2 = self._predict_risks(x, times, 'both')
            hazard_e2 = self._alpha[0] * self._lambda[0] * torch.pow(times, self._alpha[0] - 1) * risks_e2
            hazard_c2 = self._alpha[1] * self._lambda[1] * torch.pow(times, self._alpha[1] - 1) * risks_c2
            return hazard_e2, hazard_c2
        elif target == 'e':
            risks_e2 = self._predict_risks(x, times, 'e')
            return self._alpha[0] * self._lambda[0] * torch.pow(times, self._alpha[0] - 1) * risks_e2
        elif target == 'c':
            risks_c2 = self._predict_risks(x, times, 'c')
            return self._alpha[1] * self._lambda[1] * torch.pow(times, self._alpha[1] - 1) * risks_c2
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_cum_hazard(self, x, times, target='both'):
        if target == 'all':
            risks_e1, risks_e2, risks_c1, risks_c2, epsilon, gamma, kappa = self._predict_risks(x, times, 'all')
            cum_hazard_e1 = self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e1
            cum_hazard_e2 = self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e2
            cum_hazard_c1 = self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c1
            cum_hazard_c2 = self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c2
            return cum_hazard_e1, cum_hazard_e2, cum_hazard_c1, cum_hazard_c2, epsilon, gamma, kappa
        elif target == 'both':
            risks_e2, risks_c2 = self._predict_risks(x, times, 'both')
            cum_hazard_e2 = self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e2
            cum_hazard_c2 = self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c2
            return cum_hazard_e2, cum_hazard_c2
        elif target == 'e':
            risks_e2 = self._predict_risks(x, times, 'e')
            return self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e2
        elif target == 'c':
            risks_c2 = self._predict_risks(x, times, 'c')
            return self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c2
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_quantiles(self, x, quantiles, target='both'):
        assert quantiles.dim() == 1, "Quantiles should be a 1D tensor."
        output_e1, output_e2, output_c1, output_c2, _, _, _ = self.forward(x)
        risks_e1, risks_e2, risks_c1, risks_c2 = torch.exp(output_e1), torch.exp(output_e2), torch.exp(output_c1), torch.exp(output_c2)
        if target == 'both':
            q_e1 = torch.pow(-torch.log(1 - quantiles) / (self._lambda[0] * risks_e1), 1 / self._alpha[0])
            q_e2 = torch.pow(-torch.log(1 - quantiles) / (self._lambda[0] * risks_e2), 1 / self._alpha[0])
            q_c1 = torch.pow(-torch.log(1 - quantiles) / (self._lambda[1] * risks_c1), 1 / self._alpha[1])
            q_c2 = torch.pow(-torch.log(1 - quantiles) / (self._lambda[1] * risks_c2), 1 / self._alpha[1])
            return q_e1, q_e2, q_c1, q_c2
        elif target == 'both':
            q_e2 = torch.pow(-torch.log(1 - quantiles) / (self._lambda[0] * risks_e2), 1 / self._alpha[0])
            q_c2 = torch.pow(-torch.log(1 - quantiles) / (self._lambda[1] * risks_c2), 1 / self._alpha[1])
            return q_e2, q_c2
        elif target == 'e':
            return torch.pow(-torch.log(1 - quantiles) / (self._lambda[0] * risks_e2), 1 / self._alpha[0])
        elif target == 'c':
            return torch.pow(-torch.log(1 - quantiles) / (self._lambda[1] * risks_c2), 1 / self._alpha[1])
        else:
            raise ValueError(f"Invalid target: {target}")

    def reset_parameters(self):
        self._alpha.data.fill_(1.0)
        self._lambda.data.fill_(1.0)
        super(WeibullAFT_LDR, self).reset_parameters()


class LogLogisticAFT_LDR(ContinuousSurvivalBaseLDR):
    """Log-Logistic Accelerated Failure Time model via learning latent decomposed representations (LDR)."""
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
        super(LogLogisticAFT_LDR, self).__init__(
            n_features=n_features,
            rep_dims=rep_dims,
            event_dims=event_dims,
            censor_dims=censor_dims,
            norm=norm,
            activation=activation,
            dropout=dropout,
            ipm=ipm,
            alpha=alpha,
            beta=beta,
            time_grids_e=time_grids_e,
            time_grids_c=time_grids_c
        )
        # training parameters: 2 sets of (shape, scale) parameters, and network weights
        # the first set is for event, and the second set is for censoring
        self._alpha = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))
        self._lambda = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))

    def _predict_survival(self, x, times, target='both'):
        if target == 'all':
            risks_e1, risks_e2, risks_c1, risks_c2, epsilon, gamma, kappa = self._predict_risks(x, times, 'all')
            survival_e1 = 1 / (1 + self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e1)
            survival_e2 = 1 / (1 + self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e2)
            survival_c1 = 1 / (1 + self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c1)
            survival_c2 = 1 / (1 + self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c2)
            return survival_e1, survival_e2, survival_c1, survival_c2, epsilon, gamma, kappa
        elif target == 'both':
            risks_e2, risks_c2 = self._predict_risks(x, times, 'both')
            survival_e2 = 1 / (1 + self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e2)
            survival_c2 = 1 / (1 + self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c2)
            return survival_e2, survival_c2
        elif target == 'e':
            risks_e2 = self._predict_risks(x, times, 'e')
            return 1 / (1 + self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e2)
        elif target == 'c':
            risks_c2 = self._predict_risks(x, times, 'c')
            return 1 / (1 + self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c2)
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_pdf(self, x, times, target='both'):
        if target == 'all':
            survival_e1, survival_e2, survival_c1, survival_c2, epsilon, gamma, kappa = self._predict_survival(x, times,
                                                                                                               'all')
            hazard_e1, hazard_e2, hazard_c1, hazard_c2, _, _, _ = self._predict_hazard(x, times, 'all')
            return hazard_e1 * survival_e1, hazard_e2 * survival_e2, hazard_c1 * survival_c1, hazard_c2 * survival_c2, epsilon, gamma, kappa
        elif target == 'both':
            survival_e2, survival_c2 = self._predict_survival(x, times, 'both')
            hazard_e2, hazard_c2 = self._predict_hazard(x, times, 'both')
            return hazard_e2 * survival_e2, hazard_c2 * survival_c2
        elif target == 'e':
            survival_e2 = self._predict_survival(x, times, 'e')
            hazard_e2 = self._predict_hazard(x, times, 'e')
            return hazard_e2 * survival_e2
        elif target == 'c':
            survival_c2 = self._predict_survival(x, times, 'c')
            hazard_c2 = self._predict_hazard(x, times, 'c')
            return hazard_c2 * survival_c2
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_hazard(self, x, times, target='both'):
        if target == 'all':
            risks_e1, risks_e2, risks_c1, risks_c2, epsilon, gamma, kappa = self._predict_risks(x, times, 'all')
            hazard_e1 = (self._lambda[0] * self._alpha[0] * torch.pow(times, self._alpha[0] - 1) * risks_e1 /
                        (1 + self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e1))
            hazard_e2 = (self._lambda[0] * self._alpha[0] * torch.pow(times, self._alpha[0] - 1) * risks_e2 /
                        (1 + self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e2))
            hazard_c1 = (self._lambda[1] * self._alpha[1] * torch.pow(times, self._alpha[1] - 1) * risks_c1 /
                        (1 + self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c1))
            hazard_c2 = (self._lambda[1] * self._alpha[1] * torch.pow(times, self._alpha[1] - 1) * risks_c2 /
                        (1 + self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c2))
            return hazard_e1, hazard_e2, hazard_c1, hazard_c2, epsilon, gamma, kappa
        elif target == 'both':
            risks_e2, risks_c2 = self._predict_risks(x, times, 'both')
            hazard_e2 = (self._lambda[0] * self._alpha[0] * torch.pow(times, self._alpha[0] - 1) * risks_e2 /
                        (1 + self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e2))
            hazard_c2 = (self._lambda[1] * self._alpha[1] * torch.pow(times, self._alpha[1] - 1) * risks_c2 /
                        (1 + self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c2))
            return hazard_e2, hazard_c2
        elif target == 'e':
            risks_e2 = self._predict_risks(x, times, 'e')
            return (self._lambda[0] * self._alpha[0] * torch.pow(times, self._alpha[0] - 1) * risks_e2 /
                        (1 + self._lambda[0] * torch.pow(times, self._alpha[0]) * risks_e2))
        elif target == 'c':
            risks_c2 = self._predict_risks(x, times, 'c')
            return (self._lambda[1] * self._alpha[1] * torch.pow(times, self._alpha[1] - 1) * risks_c2 /
                        (1 + self._lambda[1] * torch.pow(times, self._alpha[1]) * risks_c2))
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_cum_hazard(self, x, times, target='both'):
        if target == 'all':
            survival_e1, survival_e2, survival_c1, survival_c2, epsilon, gamma, kappa = self._predict_survival(x, times,
                                                                                                               'all')
            return (-torch.log(1 - survival_e1), -torch.log(1 - survival_e2), -torch.log(1 - survival_c1),
                    -torch.log(1 - survival_c2), epsilon, gamma, kappa)
        elif target == 'both':
            survival_e2, survival_c2 = self._predict_survival(x, times, 'both')
            return -torch.log(1 - survival_e2), -torch.log(1 - survival_c2)
        elif target == 'e':
            survival_e2 = self._predict_survival(x, times, 'e')
            return -torch.log(1 - survival_e2)
        elif target == 'c':
            survival_c2 = self._predict_survival(x, times, 'c')
            return -torch.log(1 - survival_c2)
        else:
            raise ValueError(f"Invalid target: {target}")

    def _predict_quantiles(self, x, quantiles, target='both'):
        assert quantiles.dim() == 1, "Quantiles should be a 1D tensor."
        output_e1, output_e2, output_c1, output_c2, epsilon, gamma, kappa = self.forward(x)
        risks_e1, risks_e2, risks_c1, risks_c2 = torch.exp(output_e1), torch.exp(output_e2), torch.exp(output_c1), torch.exp(output_c2)
        if target == 'all':
            q_e1 = torch.pow(quantiles / ((1 - quantiles) * self._lambda[0] * risks_e1), 1 / self._alpha[0])
            q_e2 = torch.pow(quantiles / ((1 - quantiles) * self._lambda[0] * risks_e2), 1 / self._alpha[0])
            q_c1 = torch.pow(quantiles / ((1 - quantiles) * self._lambda[1] * risks_c1), 1 / self._alpha[1])
            q_c2 = torch.pow(quantiles / ((1 - quantiles) * self._lambda[1] * risks_c2), 1 / self._alpha[1])
            return q_e1, q_e2, q_c1, q_c2, epsilon, gamma, kappa
        elif target == 'both':
            q_e2 = torch.pow(quantiles / ((1 - quantiles) * self._lambda[0] * risks_e2), 1 / self._alpha[0])
            q_c2 = torch.pow(quantiles / ((1 - quantiles) * self._lambda[1] * risks_c2), 1 / self._alpha[1])
            return q_e2, q_c2
        elif target == 'e':
            return torch.pow(quantiles / ((1 - quantiles) * self._lambda[0] * risks_e2), 1 / self._alpha[0])
        elif target == 'c':
            return torch.pow(quantiles / ((1 - quantiles) * self._lambda[1] * risks_c2), 1 / self._alpha[1])
        else:
            raise ValueError(f"Invalid target: {target}")

    def reset_parameters(self):
        self._alpha.data.fill_(1.0)
        self._lambda.data.fill_(1.0)
        super(LogLogisticAFT_LDR, self).reset_parameters()
