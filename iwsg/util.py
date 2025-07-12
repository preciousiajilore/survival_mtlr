import torch

# local
from iwsg.catdist import DiscreteDist


EPS_ = 1e-4
CLIP_MIN_ = 0.001

def isnan(x):
    return torch.any(torch.isnan(x))


def safe_log(x, eps):
    return (x + eps).log()


def clip(prob, clip_min):
    return prob.clamp(min=clip_min)


class Meter:
    def __init__(self):
        self.N = 0
        self.total = 0

    def update(self, val, N):
        self.total += val
        self.N += N

    def avg(self):
        return round(self.total / self.N, 4)


def X_to_dist(X, model):
    pred_params = model(X)
    device = pred_params.device
    return DiscreteDist(logits=pred_params, n_bins=model.output_size, device=device)


def X_to_FG_dists(X, Fmodel, Gmodel):
    Fdist = X_to_dist(X, Fmodel)
    Gdist = X_to_dist(X, Gmodel)
    return Fdist, Gdist


############################################
############ IPCW BS and BLL ###############
############################################


def IPCW_batch(fn, k, tgt, Fdist, Gdist, is_g=False, detach=True):
    if is_g:
        numer_dist = Gdist
        denom_dist = Fdist
    else:
        numer_dist = Fdist
        denom_dist = Gdist

    U, Delta = tgt
    kbatch = torch.ones_like(U) * k

    ncdf_k = numer_dist.leq(kbatch)
    observed = ~Delta if is_g else Delta

    if fn == 'bll_game':
        left_loss = -1.0 * safe_log(ncdf_k, EPS_)
        right_loss = -1.0 * safe_log(1. - ncdf_k, EPS_)
    elif fn == 'bs_game':
        left_loss = (1. - ncdf_k).pow(2)
        right_loss = ncdf_k.pow(2)
    else:
        assert False

    left_numer = left_loss * observed * (U <= kbatch)

    if is_g:
        left_denom = denom_dist.gt(U)
    else:
        left_denom = denom_dist.geq(U)
    left_denom = clip(left_denom, CLIP_MIN_)

    right_numer = right_loss * (U > kbatch)
    right_denom = clip(denom_dist.gt(kbatch), CLIP_MIN_)

    if detach:
        left_denom = left_denom.detach()
        right_denom = right_denom.detach()

    left = left_numer / left_denom
    right = right_numer / right_denom
    ipcw_loss = (left + right).mean(0)
    return ipcw_loss


def uncensored_BS_or_BLL_batch(fn, k, U, Fdist):
    kbatch = torch.ones_like(U) * k
    Fk = Fdist.cdf(kbatch)
    if fn == 'bs_game':
        # BS(k) = E_T  [  1[T <= k] * (1-F(k))^2 + F(k)^2 1[T>k]  ]
        loss_k = torch.where(U <= kbatch, (1 - Fk).pow(2), Fk.pow(2))
    else:
        # BS(k) = E_T  [  1[T <= k] * (1-F(k))^2 + F(k)^2 1[T>k]  ]
        loss_k = -1.0 * torch.where(U <= kbatch, safe_log(Fk, EPS_), safe_log(1 - Fk, EPS_))

    assert loss_k.shape[0] == U.shape[0]
    loss_k = loss_k.mean(0)
    return loss_k


def cond_bs_game(fn, phase, loader, Fmodel, Gmodel, Foptimizer=None, Goptimizer=None, mode='normal', device='cuda'):
    Fsumm = 0.0
    Gsumm = 0.0
    K = Fmodel.output_size
    for k in range(K-1):
        floss_meter_k = Meter()
        gloss_meter_k = Meter()
        for batch_idx, batch in enumerate(loader):
            (U,Delta,X) = batch
            U=U.to(device)
            Delta=Delta.to(device)
            X=X.to(device)
            bsz = U.shape[0]
            if phase=='train':
                Foptimizer.zero_grad()
                Goptimizer.zero_grad()
                Fdist,Gdist = X_to_FG_dists(X, Fmodel, Gmodel)
            else:
                Fdist,Gdist = X_to_FG_dists(X, Fmodel, Gmodel)
            if mode=='normal':
                floss_k = IPCW_batch(fn, k, (U,Delta), Fdist, Gdist, is_g=False, detach=True)
                gloss_k = IPCW_batch(fn, k, (U,Delta), Fdist, Gdist, is_g=True, detach=True)
            elif mode=='uncensored':
                Fdist = X_to_dist(X, Fmodel)
                floss_k = uncensored_BS_or_BLL_batch(fn, k, U, Fdist)
                gloss_k = torch.tensor([-1.0])
            # elif mode=='kmG':
            #     assert phase=='test'
            #     Fdist = X_to_dist(X, Fmodel)
            #     G_cdfvals = _km.get_KM_cdfvals(loader, args)
            #     Gdist = _km.cdfvals_to_dist(G_cdfvals, bsz, args)
            #     floss_k = IPCW_batch(fn, k, (U,Delta), Fdist, Gdist, is_g=False, detach=True)
            #     gloss_k = torch.tensor([-1.0]).to(device)
            else:
                assert False
            if phase=='train':
                floss_k.backward()
                Foptimizer.step()
                gloss_k.backward()
                Goptimizer.step()
            floss_meter_k.update(val = floss_k.item() *  bsz, N = bsz)
            gloss_meter_k.update(val = gloss_k.item() *  bsz, N = bsz)
        Fsumm += floss_meter_k.avg()
        Gsumm += gloss_meter_k.avg()
    Fsumm = Fsumm / (K-1)
    Gsumm = Gsumm / (K-1)
    return Fsumm,Gsumm
