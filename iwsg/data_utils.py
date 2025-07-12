import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, U, Delta, X):
        self.X=X
        self.U=U
        self.Delta=Delta

    def __getitem__(self, index):
        u=self.U[index]
        delta=self.Delta[index]
        x=self.X[index]
        return u,delta,x

    def __len__(self):
        return len(self.U)


def make_loaders_from_df(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        bins: torch.Tensor,
        batch_size: int):
    n_val = df_val.shape[0]
    trainset = df2dataset(df_train, bins)
    valset = df2dataset(df_val, bins)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=n_val, shuffle=False)
    return trainloader,valloader


def tensors_to_dataset(X, U, Delta, phase, N_train=1024):
    N_full = X.shape[0]
    if phase == 'train':
        N = N_train
    elif phase in ['val', 'test']:
        N = N_full
    else:
        raise ValueError("Invalid phase: {}".format(phase))
    X = X.float()
    U = U.long()
    Delta = Delta.bool()
    dataset = SyntheticDataset(U=U[:N], Delta=Delta[:N], X=X[:N])
    return dataset


def df2dataset(
        data: pd.DataFrame,
        bins: torch.Tensor,
):
    # df_u = df['duration']
    # u = torch.tensor(df_u.to_numpy())
    delta = torch.tensor(data['event'].values).bool()
    x = torch.tensor(data.drop(columns=['time', 'event']).values).float()

    times = data['time'].values
    times = np.clip(times, 0, bins.max())
    times = torch.bucketize(times, bins, right=True)

    dataset = tensors_to_dataset(x, times, delta, phase='train', N_train=x.shape[0])
    return dataset


