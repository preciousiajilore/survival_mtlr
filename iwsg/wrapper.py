import pandas as pd
import torch
from tqdm import trange

# local
from iwsg.data_utils import make_loaders_from_df
from iwsg.util import cond_bs_game
from iwsg.catdist import DiscreteDist


def train_iwsg(
        data_train: pd.DataFrame,
        data_val: pd.DataFrame,
        Fmodel: torch.nn.Module,
        Gmodel: torch.nn.Module,
        bins: torch.Tensor,
        optimizer: str,
        batch_size: int,
        epochs: int,
        lr: float,
        lr_min: float,
        weight_decay: float,
        device: torch.device,
        early_stop: bool = True,
        patience: int = 50,
        fname: str = '',
        verbose: bool = True
):
    # convert datasets
    trainloader, valloader = make_loaders_from_df(data_train, data_val, bins, batch_size=batch_size)

    # move to device
    Fmodel = Fmodel.to(device)
    Gmodel = Gmodel.to(device)

    optimizer = getattr(torch.optim, optimizer)
    Foptimizer = optimizer(Fmodel.parameters(), lr=lr, weight_decay=weight_decay)
    Goptimizer = optimizer(Gmodel.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float('inf')
    best_ep = -1
    # training and evaluation
    prefix = f'Training w Early Stop on {device}' if early_stop else f'Training on {device} w/o Early Stop'
    pbar = trange(epochs, disable=not verbose, desc=prefix)

    for ep in pbar:
        Fmodel.train()
        Gmodel.train()
        floss_tr,gloss_tr = cond_bs_game('bll_game', 'train', trainloader, Fmodel, Gmodel, Foptimizer, Goptimizer, device=device)
        totolloss_tr = floss_tr + gloss_tr
        postfix = f"Train Loss: {totolloss_tr:.4f}"
        Fmodel.eval()
        Gmodel.eval()
        with torch.no_grad():
            if early_stop and not data_val.empty:
                floss_va, gloss_va = cond_bs_game('bll_game', 'val', valloader, Fmodel, Gmodel, device=device)
                totalloss_va = floss_va + gloss_va
                postfix += f" Val loss = {totalloss_va:.4f};"

                if best_loss > totalloss_va:
                    best_loss = totalloss_va
                    best_ep = ep
                    torch.save(Fmodel.state_dict(), fname + 'Fmodel.pth')
                    torch.save(Gmodel.state_dict(), fname + 'Gmodel.pth')
                if (ep - best_ep) > patience:
                    postfix += f' Early Stop at {ep}. Best epoch: {best_ep}. Start testing...'
                    pbar.set_postfix_str(postfix)
                    break
            pbar.set_postfix_str(postfix)
        Fmodel.load_state_dict(torch.load(fname + 'Fmodel.pth'))
        Gmodel.load_state_dict(torch.load(fname + 'Gmodel.pth'))

    return Fmodel, Gmodel


def make_survival_prediction(
        X: torch.Tensor,
        n_bins: int,
        model: torch.nn.Module,
        device: torch.device
):
    model.eval()
    with torch.no_grad():
        pred_logits = model(X)
        dist = DiscreteDist(pred_logits, n_bins, device=device)
        pred_survival = dist.predict_cond_survival_dist()
    return pred_survival



