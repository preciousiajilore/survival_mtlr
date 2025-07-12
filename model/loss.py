import torch
import torch.nn as nn
from utils import safe_log, safe_sqrt
from utils.util_survival import KaplanMeierTorch

def masked_logsumexp(
        x: torch.Tensor,
        mask: torch.Tensor,
        dim: int=-1
) -> torch.Tensor:
    """Computes logsumexp over elements of a tensor specified by a mask (two-level)
    in a numerically stable way.

    :param x: torch.Tensor, input tensor, shape (n_batch, n_features)
    :param mask: torch.Tensor, mask tensor, shape (n_batch, n_features)
        1s in positions that should be used for logsumexp computation and 0s everywhere else.
    :param dim: int, dimension to sum over
        The dimension of `x` over which logsumexp is computed. Default -1 uses the last dimension.
    :return: torch.Tensor, logsumexp over elements of a tensor specified by a mask
    """
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return safe_log(torch.sum(torch.exp((x - max_val.unsqueeze(dim)) * mask) * mask, dim=dim)) + max_val


def crossing_loss(
        y_pred: torch.Tensor
):
    """
    Crossing loss for quantile regression, where adjacent quantiles are consecutive
    https://stats.stackexchange.com/questions/249874/the-issue-of-quantile-curves-crossing-each-other
    :param y_pred:  torch.Tensor, predicted quantiles, shape (n_batch, n_quantiles)
    :return: torch.Tensor, crossing loss
    """
    margin=0.1
    alpha=10
    diffs = y_pred[:, 1:] - y_pred[:, :-1] # we would like diffs all to be +ve if not crossing
    loss_cross = alpha*torch.mean(torch.maximum(torch.tensor(0.0), margin -diffs))
    return loss_cross


def quantile_loss(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        cen_indicator: torch.Tensor,
        taus: torch.Tensor
):
    """
    Standard checkmark / tilted pinball loss used for quantile regression,
    but we also pass in cen_indicator and avoid calculating this over those datapoints

    The code is borrowed from https://github.com/TeaPearce/Censored_Quantile_Regression_NN
    :param y_pred: torch.Tensor, predicted quantiles, shape (n_batch, n_quantiles)
    :param y_true: torch.Tensor, true event times, shape (n_batch, 1)
    :param cen_indicator: torch.Tensor, censoring indicator, shape (n_batch, 1)
    :param taus: torch.Tensor, quantile levels, shape (n_quantiles,)
    :return:
    """
    tau_block = taus.repeat((cen_indicator.shape[0], 1))  # need this stacked in shape (n_batch, n_quantiles)
    loss = torch.sum((cen_indicator < 1) * (y_pred - y_true) * ((1 - tau_block) - 1. * (y_pred < y_true)), dim=1)
    loss = torch.mean(loss)
    return loss


class PartialLikelihood(nn.Module):
    """Computes the partial likelihood loss for CoxPH model."""

    def __init__(
            self,
            reduction: str="mean"
    ):
        super(PartialLikelihood, self).__init__()
        assert reduction in ["mean", "sum"], "reduction must be one of 'mean', 'sum'"
        self.reduction = reduction

    def forward(
            self,
            risk_pred: torch.Tensor,
            y_true: torch.Tensor
    ):
        t_true, e_true = y_true[:, 0], y_true[:, 1]
        # check whether e_true is all 0
        if e_true.sum() == 0:
            return torch.tensor(0.0).to(risk_pred.device)
        else:
            risk_pred = risk_pred.reshape(-1, 1)
            t_true = t_true.reshape(-1, 1)
            e_true = e_true.reshape(-1, 1)
            mask = torch.ones(t_true.shape[0], t_true.shape[0]).to(t_true.device)
            mask[(t_true.T - t_true) > 0] = 0
            max_risk = risk_pred.max()
            log_loss = torch.exp(risk_pred - max_risk) * mask
            log_loss = torch.sum(log_loss, dim=0)
            log_loss = safe_log(log_loss).reshape(-1, 1) + max_risk
            # Sometimes in the batch we got all censoring data, so the denominator gets 0 and throw nan.
            # Solution: Consider increase the batch size. After all the nll should be performed on the whole dataset.
            # Based on equation 2&3 in https://arxiv.org/pdf/1606.00931.pdf
            nll = -torch.sum((risk_pred - log_loss) * e_true) / torch.sum(e_true)

            if self.reduction == "mean":
                nll = nll / risk_pred.shape[0]
            elif self.reduction == "sum":
                nll = nll

            return nll


class LikelihoodMTLR(nn.Module):
    """Computes the negative log-likelihood for MTLR model."""

    def __init__(
            self,
            reduction: str="mean"
    ):
        super(LikelihoodMTLR, self).__init__()
        assert reduction in ["mean", "sum"], "reduction must be one of 'mean', 'sum'"
        self.reduction = reduction

    def forward(
            self,
            logits: torch.Tensor,
            encoded_target: torch.Tensor
    ):
        censored = encoded_target.sum(dim=1) > 1
        nll_censored = masked_logsumexp(logits[censored], encoded_target[censored]).sum() if censored.any() else 0
        nll_uncensored = (logits[~censored] * encoded_target[~censored]).sum() if (~censored).any() else 0

        # the normalising constant
        norm = torch.logsumexp(logits, dim=1).sum()

        nll_total = -(nll_censored + nll_uncensored - norm)
        if self.reduction == "mean":
            nll_total = nll_total / encoded_target.size(0)
        elif self.reduction == "sum":
            nll_total = nll_total

        return nll_total

class Likelihood(nn.Module):
    """Computes the negative log-likelihood for continuous survival model."""

    def __init__(
            self,
            reduction: str="mean"
    ):
        super(Likelihood, self).__init__()
        assert reduction in ["mean", "sum"], "reduction must be one of 'mean', 'sum'"
        self.reduction = reduction

    def forward(
            self,
            survival: torch.Tensor,
            pdf: torch.Tensor,
            y_true: torch.Tensor
    ):
        t_true, e_true = y_true[:, 0], y_true[:, 1]
        loglikelihood = e_true * safe_log(pdf) + (1 - e_true) * safe_log(survival)
        nll = -loglikelihood
        if self.reduction == "mean":
            nll = nll.mean(dim=-1)
        elif self.reduction == "sum":
            nll = nll.sum(dim=-1)
        return nll


class CensoredPinballLoss(nn.Module):
    """Computes the censored pinball loss for quantile regression."""

    def __init__(
            self,
            quantiles: torch.Tensor,
            use_cross_loss: bool = False,
            reduction: str = "mean"
    ):
        super(CensoredPinballLoss, self).__init__()
        self.quan_levels = quantiles.reshape([1, -1])
        self.reduction = reduction
        self._t_max = None
        self.IS_USE_CROSS_LOSS = use_cross_loss

    @property
    def t_max(self):
        return self._t_max

    @t_max.setter
    def t_max(self, value):
        print("Setting t_max to {} for CQRNN.".format(value))
        self._t_max = value

    def forward(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor
    ):
        t_true, e_true = y_true[:, 0], y_true[:, 1]
        t_true = t_true.reshape(-1, 1)
        e_true = e_true.reshape(-1, 1)
        c_true = 1 - e_true
        self.quan_levels = self.quan_levels.to(y_true.device)

        # y_pred is shape (n_batch, n_quantiles)
        # y_true is shape (n_batch, 1)
        # cen_indicator is shape (n_batch, 1)
        # use detach to figure out where to block
        # first figure out closest quantile (do for all observations)
        y_pred_detach = y_pred.detach()
        # do we need detach()? yes I think so, otherwise loss is affected, though it's argmin so gradients prob don't flow anyway

        # should do this outside loss really and subselect here if needed
        quan_level_block = self.quan_levels.repeat((c_true.shape[0], 1)).to(y_true.device)

        loss_obs = quantile_loss(y_pred, t_true, c_true, self.quan_levels)

        # add in crossing loss
        if self.IS_USE_CROSS_LOSS:
            loss_obs += crossing_loss(y_pred)

        # use argmin to get nearest quantile
        torch_abs = torch.abs(t_true - y_pred_detach[:, :])
        estimated_quantiles = torch.max(
            quan_level_block[:, :] * (
                        torch_abs == torch.min(torch_abs, dim=1).values.view(torch_abs.shape[0], 1)), dim=1).values

        # compute weights, eq 11, Stephen Portnoy 2003
        # want weights to be in shape (batch_size x n_quantiles-1)
        weights = (quan_level_block[:, :] < estimated_quantiles.reshape(-1, 1)) * 1. + (
                quan_level_block[:, :] >= estimated_quantiles.reshape(-1, 1)) * (
                          quan_level_block[:, :] - estimated_quantiles.reshape(-1, 1)) / (
                          1 - estimated_quantiles.reshape(-1, 1))

        # now compute censored loss using
        # weight * censored value, + (1-weight) * fictionally large value
        loss_cens = torch.sum((c_true > 0) *
                              (weights * (y_pred[:, :] - t_true) * (
                                      (1 - quan_level_block[:, :]) - 1. * (y_pred[:, :] < t_true)) +
                               (1 - weights) * (y_pred[:, :] - self.t_max) * (
                                       (1 - quan_level_block[:, :]) - 1. * (y_pred[:, :] < self.t_max)))
                              , dim=1)
        loss_cens = torch.mean(loss_cens)

        return loss_obs + loss_cens


class OrthoNets(nn.Module):
    """ Orthogonal Regularization for Hidden Weights."""

    def __init__(
            self,
            input_size: int = 1,
            alpha: float=1e-4,
    ):
        super(OrthoNets, self).__init__()
        self.alpha = alpha
        self.start_weight = torch.eye(input_size)

    def forward(
            self,
            model: nn.Module
    ):
        w_eps_mul = self.start_weight
        for name, param in model.epsilon_net.named_parameters():
            if 'weight' in name and 'linear' in name:
                w_eps_mul = torch.mm(param, w_eps_mul)
        # w_eps_mean = torch.mean(w_eps_mul, dim=0)
        w_eps_mean = torch.mean(torch.abs(w_eps_mul), dim=0)

        w_del_mul = self.start_weight
        for name, param in model.kappa_net.named_parameters():
            if 'weight' in name and 'linear' in name:
                w_del_mul = torch.mm(param, w_del_mul)
        # w_del_mean = torch.mean(w_del_mul, dim=0)
        w_del_mean = torch.mean(torch.abs(w_del_mul), dim=0)

        w_gam_mul = self.start_weight
        for name, param in model.gamma_net.named_parameters():
            if 'weight' in name and 'linear' in name:
                w_gam_mul = torch.mm(param, w_gam_mul)
        # w_gam_mean = torch.mean(w_gam_mul, dim=0)
        w_gam_mean = torch.mean(torch.abs(w_gam_mul), dim=0)

        # orthogonal_loss = (torch.mean(torch.abs(w_eps_mean * w_del_mean)) +
        #                    torch.mean(torch.abs(w_eps_mean * w_gam_mean)) +
        #                    torch.mean(torch.abs(w_del_mean * w_gam_mean)))
        orthogonal_loss = (torch.mean(w_eps_mean * w_del_mean) +
                            torch.mean(w_eps_mean * w_gam_mean) +
                            torch.mean(w_del_mean * w_gam_mean))
        return self.alpha * orthogonal_loss

    def to(self, *args, **kwargs):
        self.start_weight = self.start_weight.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class OrthoReps(nn.Module):
    # Orthogonal Regularization (Cosine similarity) for Representations

    def __init__(
            self,
            alpha: float=1e-4,
    ):
        super(OrthoReps, self).__init__()
        self.alpha = alpha

    def forward(
            self,
            reps: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Calculate the Orthogonal Regularization for Representations.
        :param reps: list of representations
            Each representation should have the shape of (n_batch, n_dims).
        :return: torch.Tensor, Orthogonal Regularization for Representations.
        """
        ortho_loss = 0
        for i in range(len(reps)):
            reps_1 = reps[i]
            for j in range(i + 1, len(reps)):
                reps_2 = reps[j]
                # cosine similarity between the two groups of representations
                cos_sim = torch.nn.functional.cosine_similarity(reps_1, reps_2, dim=1)
                ortho_loss += torch.mean(cos_sim)
        return self.alpha * ortho_loss


class IPM(nn.Module):
    """
    Integral probability metrics (IPM) loss to calculate the discrepancy of the empirical distribution of
    confounder representation (kappa) between the event and censored groups.
    """

    def __init__(
            self,
            method: str= 'mmd-linear',
            alpha: float=1.0,
    ):
        """
        Initialize the IPM loss.
        :param method: str, method to calculate IPM loss, one of ['mmd-linear', 'mmd-rbf', 'wasserstein',
            'mmd2-linear', 'mmd2-rbf', 'wasserstein2'].
        :param alpha: float, weight for the IPM loss.
        """
        super(IPM, self).__init__()
        self.alpha = alpha
        self.gamma = None
        self.method = method
        self.km_e = None
        self.km_c = None
        self.fitted = False

    def fit(
            self,
            event_times: torch.Tensor,
            event_indicators: torch.Tensor,
            device: torch.device
    ):
        event_times, event_indicators = event_times.to(device), event_indicators.to(device)
        self.km_e = KaplanMeierTorch(event_times, event_indicators)
        self.km_c = KaplanMeierTorch(event_times, 1 - event_indicators)
        self.fitted = True

    def forward(
            self,
            rep: torch.Tensor,
            t_batch: torch.Tensor,
            e_batch: torch.Tensor,
            event: bool = True
    ):
        """
        Calculate the discrepancy of the empirical distribution of confounder representation (kappa) between the event
        and censored groups.
        :param rep: Tensor, confounder representations.
        :param t_batch: Tensor, event time.
        :param e_batch: Tensor, event indicator.
        :param event: bool, whether to calculate the IPM loss for the event group.
        :return: Tensor, discrepancy of the empirical distribution of confounder representation (kappa) between the event
        and censored groups.
        """
        if not self.fitted:
            raise ValueError("IPM loss must be fitted first.")

        km = self.km_e if event else self.km_c
        surv_prob_at_t = km.predict(t_batch)
        # get weights with the rules:
        # if event_indicators == 1, and surv_prob_at_event > 0.5 then weight = 1
        # if event_indicators == 1, and surv_prob_at_event <= 0.5 then weight = 0
        # if event_indicators == 0, and surv_prob_at_event > 0.5 then weight = (a - 0.5)/ a
        # if event_indicators == 0, and surv_prob_at_event <= 0.5 then weight = 0
        weights = torch.where((e_batch == 1) & (surv_prob_at_t > 0.5), 1.0, 0.0)
        weights += torch.where((e_batch == 0) & (surv_prob_at_t > 0.5), (surv_prob_at_t - 0.5) / surv_prob_at_t, 0.0)

        if self.method in ['mmd2-lin', 'mmd-lin']:
            rep_1_avg = rep.T @ weights / weights.sum()
            rep_2_avg = rep.T @ (1 - weights) / (1 - weights).sum()
            ipm = torch.sum((rep_1_avg - rep_2_avg) ** 2)
            if self.method == 'mmd-lin':
                ipm = safe_sqrt(ipm)
            ipm = self.alpha * ipm
        elif self.method in ['mmd2-rbf', 'mmd-rbf']:
            if self.gamma is None:  # set gamma using median heuristic at the first batch
                sigma = median_heuristic(rep)
                self.gamma = 1 / (2 * sigma ** 2)

            rep_1 = rep[weights != 0]
            rep_2 = rep[weights != 1]

            weights_1 = weights[weights != 0] / torch.sum(weights)
            weights_2 = (1 - weights[weights != 1]) / torch.sum(1 - weights)
            weight_mat_11 = torch.outer(weights_1, weights_1)
            weight_mat_12 = torch.outer(weights_1, weights_2)
            weight_mat_22 = torch.outer(weights_2, weights_2)

            k11 = rbf_kernel(rep_1, rep_1, gamma=self.gamma)
            k12 = rbf_kernel(rep_1, rep_2, gamma=self.gamma)
            k22 = rbf_kernel(rep_2, rep_2, gamma=self.gamma)

            ipm = torch.sum(weight_mat_11 * k11) + torch.sum(weight_mat_22 * k22) - 2 * torch.sum(weight_mat_12 * k12)

            if self.method == 'mmd-rbf':
                ipm = safe_sqrt(ipm)
            ipm = self.alpha * ipm
        elif self.method in ['wasserstein', 'wasserstein2']:
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid method: {self.method}")
        return ipm


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute the RBF (Gaussian) kernel between all pairs of x and y.

    :param x: Tensor of shape [N, D] (N samples, D dimensions).
    :param y: Tensor of shape [M, D] (M samples, D dimensions).
    :param gamma: Kernel width. gamma = 1 / (2 * sigma ** 2)
    Returns:
        torch.Tensor: A [N, M] tensor of RBF kernel values.
    """
    # Pairwise squared distances
    # shape: [N, M]
    pairwise_sq_dists = torch.cdist(x, y, p=2).pow(2)

    # RBF kernel
    kxy = torch.exp(-pairwise_sq_dists * gamma)
    return kxy


def median_heuristic(x: torch.Tensor) -> float:
    """
    The default heuristic for the bandwidth of the RBF kernel.

    1. Randomly sample a subset of points from your data.
    2. Compute the pairwise distances for that subset.
    3. Take the median of those distances as the bandwidth (sigma).

    Here, because the IPM calculation is done on each batch, which can be considered as a subset of the whole data,
     we can use the median of the pairwise distances for that batch as the bandwidth.
    """
    # Compute pairwise distances
    pairwise_dists = torch.cdist(x, x, p=2)
    # Get the median of all pairwise distances
    median_val = pairwise_dists.median()
    return median_val.item()
