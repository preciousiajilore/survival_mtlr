import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import trange
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import wandb
import torchtuples as tt
from feature_selection import unicox_fs, coxen_fs
# models
from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
from model.Survival import MTLR, CoxPH, CenQuanRegNN, WeibullAFT, LogLogisticAFT
from model.LatDecReps import MTLR_LDR, CoxPH_LDR, WeibullAFT_LDR, LogLogisticAFT_LDR

from pycox.models import DeepHitSingle, CoxTime, LogisticHazard
from pycox.models.cox_time import MLPVanillaCoxTime

from iwsg.models import DiscNN
from iwsg.wrapper import train_iwsg, make_survival_prediction

from utils import save_params, set_seed, print_performance, pad_tensor
from utils.util_survival import survival_data_split, make_time_bins, format_pred_sksurv, \
    make_mono_quantiles
from args import generate_parser
from SurvivalEVAL import SurvivalEvaluator, QuantileRegEvaluator

folder = 'logs/Baseline'
# create folder if it does not exist
if not os.path.exists(folder):
    os.makedirs(folder)

#time_coordinates = np.array(time_coordinates)
#time_coordinates = np.sort(np.unique(time_coordinates))

DISCRETE_MODELS = [
    "MTLR", "MTLR-LDR",
    "DeepHit", "Nnet-survival", "IWSG"]




def main(args=None):
    if isinstance(args, argparse.Namespace):
        wandb.init(
            project="UroLOGIC",
            config=args,
            name=args.model
        )
    else:
        wandb.init(config=args)
    wandb.define_metric("C-index", summary="mean")
    wandb.define_metric("IBS", summary="mean")
    wandb.define_metric("MAE_Hinge", summary="mean")
    wandb.define_metric("MAE_PO", summary="mean")
    wandb.define_metric("KM-cal", summary="mean")
    wandb.define_metric("D-cal", summary="mean")
    wandb.define_metric("BS", summary="mean")
    wandb.define_metric("AUC", summary="mean")

    args = wandb.config
    data = pd.read_csv("data/prepostop.csv")
    def bracket_to_num(val):
        if isinstance(val, str) and re.fullmatch(r"\[\d+\]", val):
            return int(val.strip("[]"))
        return val
    
    
     
    
    data.replace('?', np.nan, inplace=True)
    data.rename(columns={'time_to_event': 'time',
                         'failure': 'event'}, inplace=True)
    # columns that need to be standardized
    """
      Index(['distal', 'penile', 
       '#strictures', 'charlsons', 'cormorbidity', 'diabetes', 'copd',
       'smoker', 'bmi35+', 'prevprocedure', '#prevprocedures',  'open',
       'urine', 'stx_length_1', 'stx_length_2', 'stxetiology_0',
       'stxetiology_1', 'stxetiology_2', 'stxetiology_3', 'stxetiology_4',
       'stxetiology_5', 'stxetiology_6', 'stxlocation_0', 'stxlocation_1',
       'stxlocation_2', 'stxlocation_3', 'stxlocation_4', 'stxlocation_5',
       'stxlocation_6', 'stx_length_1', 'stx_length_2', 'cysto_0.0',
       'cysto_1.0', 'cysto_2.0', 'cysto_3.0', 'abx', 'erectilepre', 'uti',
       'los(days)', 'spc', 'tissue', 'transection', 'urethroplasty',
       'cathremoval', 'cathdays', 'tissue_0.0', 'tissue_1.0', 'tissue_2.0',
       'tissue_3.0', 'tissue_4.0', 'tissue_5.0', 'urethroplasty_1.0',
       'urethroplasty_2.0', 'urethroplasty_3.0', 'urethroplasty_4.0',
       'urethroplasty_5.0', 'urethroplasty_6.0', 'failure', 'time_to_event'],
      dtype='object')
    """
    

    cols_stdz = ['stx_length_1', 'stx_length_2']
    #features = data.columns.to_list()
    
    features = [
   # drop event/time
       'distal', 'penile', 
       '#strictures', 'charlsons', 'cormorbidity', 'diabetes', 'copd',
       'smoker', 'bmi35+', 'prevprocedure', '#prevprocedures','open',
       'urine', 'stxetiology_0',
       'stxetiology_1', 'stxetiology_2', 'stxetiology_3', 'stxetiology_4',
       'stxetiology_5', 'stxetiology_6', 'stxlocation_0', 'stxlocation_1',
       'stxlocation_2', 'stxlocation_3', 'stxlocation_4', 'stxlocation_5',
       'stxlocation_6', 'stx_length_1', 'stx_length_2', 'cysto_0.0',
       'cysto_1.0', 'cysto_2.0', 'cysto_3.0', 'abx', 'erectilepre', 'uti',
       'los(days)', 'spc', 'transection',
       'tissue_0.0', 'tissue_1.0', 'tissue_2.0',
       'tissue_3.0', 'tissue_4.0', 'tissue_5.0', 'urethroplasty_1.0',
       'urethroplasty_2.0', 'urethroplasty_3.0', 'urethroplasty_4.0',
       'urethroplasty_5.0', 'urethroplasty_6.0','time','event'
   
]

    
    for col in features:
        # Only replace in object (string) columns to avoid breaking real floats
        if data[col].dtype == object:
            data[col] = data[col].str.replace(',', '.')
            print(col)

 

    

    #Castt feature values to float
    data = data[features].astype("float32")
    data = data.dropna(subset=features + ['time', 'event']).reset_index(drop=True)
    
    data = data[data['time'] >= 0].reset_index(drop=True)
    #print('NaNs per column:\n', data.isna().sum())
    #print(data[data['time'] < 0])

    cols_wo_stdz = list(set(features).symmetric_difference(cols_stdz))  # including time and event
    stdz = [([col], StandardScaler()) for col in cols_stdz]
    wo_stdz = [(col, None) for col in cols_wo_stdz]
    columns_transform = stdz + wo_stdz

    #Debugging
    print(data[features].isnull().sum())
    print(data[features].dtypes)

    if args.early_stop == False or args.model in ["CoxNet", "RSF", "GB"]:
        pct_train = 0.6
        pct_val = 0.0
        pct_test = 0.4
    else:
        pct_train = 0.4
        pct_val = 0.2
        pct_test = 0.4
    
    args.n_features = len(features) - 2     # excluding time and event
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    path = save_params(args)

    ci = []
    mae_hinge = []
    mae_po = []
    ibs = []
    km_cal = []
    d_cal = []
    bs = []
    auc = []

    pbar_outer = trange(args.n_exp, disable=not args.verbose, desc='Experiment')
    for i in pbar_outer:
        seed_i = args.seed + i
        set_seed(seed_i, device)
        data_train, data_val, data_test = survival_data_split(data, stratify_colname='both', frac_train=pct_train,
                                                              frac_val=pct_val, frac_test=pct_test, random_state=seed_i)
        # standardize the data
        # [features] to keep the order, otherwise the feature order will be changed and the result is not reproducible
        mapper_df = DataFrameMapper(columns_transform, df_out=True)
        data_train = mapper_df.fit_transform(data_train).astype('float32')[features]
        data_val = mapper_df.transform(data_val).astype('float32')[features] if not data_val.empty else data_val
        data_test = mapper_df.transform(data_test).astype('float32')[features]
        
        #feature selection
        if args.fs == "lasso_cox":
            data_train, data_val, data_test = coxen_fs(data_train, data_val, data_test, verbose=args.verbose)
        elif args.fs == "cox_score_test":
            data_train, data_val, data_test = unicox_fs(data_train, data_val, data_test, verbose=args.verbose)
        elif args.fs == "None":
            pass
        else:
            raise ValueError(f"Unknown feature selection method: {args.fs}")
        data_train.reset_index(drop=True, inplace=True)
        data_val.reset_index(drop=True, inplace=True)
        data_test.reset_index(drop=True, inplace=True)

        data_train_val = pd.concat([data_train, data_val], ignore_index=True) if not data_val.empty else data_train
        n_features = data_train.shape[1] - 2  # excluding time and event
        x_train = data_train.drop(["time", "event"], axis=1).values
        t_train, e_train = data_train["time"].values, data_train["event"].values
        x_val = data_val.drop(["time", "event"], axis=1).values if not data_val.empty else None
        t_val, e_val = data_val["time"].values, data_val["event"].values if not data_val.empty else None
        x_test = data_test.drop(['time', 'event'], axis=1).values
        t_test, e_test = data_test["time"].values, data_test["event"].values
        x_train_val = data_train_val.drop(["time", "event"], axis=1).values
        t_train_val, e_train_val = data_train_val["time"].values, data_train_val["event"].values
        """
        print("NaNs after splitting:", data_train.isnull().sum())
        print("NaNs after mapping:", data_train.isnull().sum())
        print("Check values:", data_train.describe())

        print('Sample x_train:', x_train[:5])
        print('Sample t_train:', t_train[:5])
        print('Sample e_train:', e_train[:5])
        print('t_train min/max:', np.min(t_train), np.max(t_train))
        print('e_train unique:', np.unique(e_train))
        print("Event distribution:", np.unique(e_train, return_counts=True))
        print("Time min/max:", np.min(t_train), np.max(t_train))
        """
        # create time bins for discrete survival analysis models
        if args.model in DISCRETE_MODELS:
            discrete_bins_e = make_time_bins(t_train, event=e_train)
            discrete_bins_c = make_time_bins(t_train, event=1 - e_train)
            if args.model in ["DeepHit", "Nnet-survival"]:
                # the first bin of DeepHit must smaller than the smallest time in the data
                discrete_bins_e[0] = float(max(t_train_val.min() - 1e-5, 0))
                discrete_bins_c[0] = float(max(t_train_val.min() - 1e-5, 0))

        if "CoxPH" in args.model:
            if args.model == "CoxPH":
                model = CoxPH(
                    n_features=n_features,
                    hidden_size=args.neurons,
                    norm=args.norm,
                    activation=args.activation,
                    dropout=args.dropout
                )
            elif args.model == "CoxPH-LDR":
                model = CoxPH_LDR(
                    n_features=n_features,
                    rep_dims=args.neurons,
                    event_dims=args.d_dims,
                    censor_dims=args.d_dims,    # this is for easier hyperparameter tuning,
                    norm=args.norm,
                    activation=args.activation,
                    dropout=args.dropout,
                    ipm=args.ipm,
                    alpha=args.alpha,
                    beta=args.beta,
                )
            else:
                raise ValueError(f"Unknown model name: {args.model}")
            model.fit(data_train, data_val, device=device, optimizer=args.optimizer, batch_size=args.batch_size,
                      epochs=args.n_epochs, lr=args.lr, lr_min=1e-3 * args.lr, weight_decay=args.weight_decay,
                      early_stop=args.early_stop, fname=folder + f'/{model.__class__.__name__}_{args.timestamp}',
                      verbose=args.verbose)
            x_test = torch.from_numpy(x_test).float().to(device)
            surv_test = model.predict_survival(x_test)
            time_coordinates = model.time_bins
            """
            print("time_coordinates:", time_coordinates)
            print("diffs:", np.diff(time_coordinates))
            print("duplicates:", len(time_coordinates) - len(np.unique(time_coordinates)))
            print("min/max:", time_coordinates.min(), time_coordinates.max())
            """
        elif "MTLR" in args.model:
            if args.model == "MTLR":
                model = MTLR(
                    n_features=n_features,
                    time_bins=discrete_bins_e,
                    hidden_size=args.neurons,
                    norm=args.norm,
                    activation=args.activation,
                    dropout=args.dropout
                )
            elif args.model == "MTLR-LDR":
                model = MTLR_LDR(
                    n_features=n_features,
                    time_bins_event=discrete_bins_e,
                    time_bins_censor=discrete_bins_c,
                    rep_dims=args.neurons,
                    event_dims=args.d_dims,
                    censor_dims=args.d_dims,  # this is for easier hyperparameter tuning
                    norm=args.norm,
                    activation=args.activation,
                    dropout=args.dropout,
                    ipm=args.ipm,
                    alpha=args.alpha,
                    beta=args.beta,
                )
            else:
                raise ValueError(f"Unknown model name: {args.model}")
            model.fit(data_train, data_val, device=device, optimizer=args.optimizer, batch_size=args.batch_size,
                      epochs=args.n_epochs, lr=args.lr, lr_min=1e-3 * args.lr, weight_decay=args.weight_decay,
                      early_stop=args.early_stop, fname=folder + f'/{model.__class__.__name__}_{args.timestamp}',
                      verbose=args.verbose)
            x_test = torch.from_numpy(x_test).float().to(device)
            surv_test = model.predict_survival(x_test)
            time_coordinates = model.time_bins
            if time_coordinates[0] > 0:
                time_coordinates = pad_tensor(time_coordinates, 0, where='start')
            elif time_coordinates[0] == 0:
                surv_test = surv_test[:, 1:]
            # time_coordinates = np.sort(np.unique(time_coordinates)) # why?
        elif args.model == "CQRNN":
            model = CenQuanRegNN(
                n_features=n_features,
                hidden_size=args.neurons,
                n_quantiles=9,
                norm=args.norm,
                activation=args.activation,
                dropout=args.dropout,
                t_max=1.2 * data_train.time.max()
            )
            model.fit(data_train, data_val, device=device, optimizer=args.optimizer, batch_size=args.batch_size,
                      epochs=args.n_epochs, lr=args.lr, lr_min=1e-3 * args.lr, weight_decay=args.weight_decay,
                      early_stop=args.early_stop, fname=folder + f'/{model.__class__.__name__}', verbose=args.verbose)
            x_test = torch.from_numpy(x_test).float().to(device)
            quan_test = model.predict_quantiles(x_test)
            # quan_test = pad_tensor(quan_test, 0, where='start')     # for quantile = 0, the prediction is 0
            quan_levels = model.quan_levels
            # quan_levels = pad_tensor(quan_levels, 0, where='start')
            quan_levels, quan_test = make_mono_quantiles(quan_levels.cpu().numpy(), quan_test.cpu().numpy(),
                                                         method="bootstrap", seed=seed_i)
        elif "WeibullAFT" in args.model:
            if args.model == "WeibullAFT":
                model = WeibullAFT(
                    n_features=n_features,
                    hidden_size=args.neurons,
                    norm=args.norm,
                    activation=args.activation,
                    dropout=args.dropout
                )
            elif args.model == "WeibullAFT-LDR":
                model = WeibullAFT_LDR(
                    n_features=n_features,
                    rep_dims=args.neurons,
                    event_dims=args.d_dims,
                    censor_dims=args.d_dims,    # this is for easier hyperparameter tuning
                    norm=args.norm,
                    activation=args.activation,
                    dropout=args.dropout,
                    ipm=args.ipm,
                    alpha=args.alpha,
                    beta=args.beta,
                )
            else:
                raise ValueError(f"Unknown model name: {args.model}")
            model.fit(data_train, data_val, device=device, optimizer=args.optimizer, batch_size=args.batch_size,
                      epochs=args.n_epochs, lr=args.lr, lr_min=1e-3 * args.lr, weight_decay=args.weight_decay,
                      early_stop=args.early_stop, fname=folder + f'/{model.__class__.__name__}_{args.timestamp}',
                      verbose=args.verbose)
            x_test = torch.from_numpy(x_test).float().to(device)
            surv_test = model.predict_survival(x_test)
            time_coordinates = model.t_grids
        elif "LogLogisticAFT" in args.model:
            if args.model == "LogLogisticAFT":
                model = LogLogisticAFT(
                    n_features=n_features,
                    hidden_size=args.neurons,
                    norm=args.norm,
                    activation=args.activation,
                    dropout=args.dropout
                )
            elif args.model == "LogLogisticAFT-LDR":
                model = LogLogisticAFT_LDR(
                    n_features=n_features,
                    rep_dims=args.neurons,
                    event_dims=args.d_dims,
                    censor_dims=args.d_dims,    # this is for easier hyperparameter tuning
                    norm=args.norm,
                    activation=args.activation,
                    dropout=args.dropout,
                    ipm=args.ipm,
                    alpha=args.alpha,
                    beta=args.beta,
                )
            else:
                raise ValueError(f"Unknown model name: {args.model}")
            model.fit(data_train, data_val, device=device, optimizer=args.optimizer, batch_size=args.batch_size,
                      epochs=args.n_epochs, lr=args.lr, lr_min=1e-3 * args.lr, weight_decay=args.weight_decay,
                      early_stop=args.early_stop, fname=folder + f'/{model.__class__.__name__}_{args.timestamp}',
                      verbose=args.verbose)
            x_test = torch.from_numpy(x_test).float().to(device)
            surv_test = model.predict_survival(x_test)
            time_coordinates = model.t_grids
        elif args.model == "LogNormalNN":
            # deprecated
            raise ValueError("LogNormalNN is deprecated, use other AFT models instead.")
            # model = LogNormalNN(
            #     n_features=n_features,
            #     hidden_size=args.neurons,
            #     norm=args.norm,
            #     activation=args.activation,
            #     dropout=args.dropout,
            #     lam=args.lam
            # )
            # model.fit(data_train, data_val, device=device, batch_size=args.batch_size, epochs=args.n_epochs,
            #           lr=args.lr, lr_min=1e-3 * args.lr, weight_decay=args.weight_decay, early_stop=args.early_stop,
            #           fname=folder + f'/{model.__class__.__name__}', verbose=args.verbose)
            # x_test = torch.from_numpy(x_test).float().to(device)
            # surv_test = model.predict_survival(x_test)
            # time_coordinates = model.time_bins
        elif args.model in ["DeepHit", "Nnet-survival"]:
            labtrans = DeepHitSingle.label_transform(discrete_bins_e.numpy())
            net = tt.practical.MLPVanilla(in_features=n_features, num_nodes=args.neurons,
                                          out_features=labtrans.out_features, batch_norm=args.norm,
                                          dropout=args.dropout, activation=getattr(nn, args.activation))
            optim = getattr(tt.optim, args.optimizer)
            if args.model == "DeepHit":
                model = DeepHitSingle(net, optim, device=args.device, alpha=0.2, sigma=0.1,
                                      duration_index=labtrans.cuts)
            elif args.model == "Nnet-survival":
                model = LogisticHazard(net, optim, device=args.device, duration_index=labtrans.cuts)
            model.label_transform = labtrans

            y_train = model.label_transform.transform(*(t_train, e_train))
            y_val = model.label_transform.transform(*(t_val, e_val))

            val = (x_val, y_val)
            val_size = x_val.shape[0]

            model.optimizer.set_lr(args.lr)
            model.optimizer.set('weight_decay', args.weight_decay)
            if args.early_stop:
                callbacks = [tt.callbacks.EarlyStopping()]
            else:
                callbacks = None
            model.fit(input=x_train, target=y_train, batch_size=args.batch_size, epochs=args.n_epochs,
                      callbacks=callbacks, verbose=args.verbose, val_data=val, val_batch_size=val_size)
            surv_df = model.predict_surv_df(x_test)
            time_coordinates = surv_df.index.values
            surv_test = surv_df.values.T
            time_coordinates = np.sort(np.unique(time_coordinates))
        elif args.model == "CoxTime":
            labtrans = CoxTime.label_transform()
            labtrans.fit(t_train, e_train)
            net = MLPVanillaCoxTime(in_features=n_features, num_nodes=args.neurons, batch_norm=args.norm,
                                    dropout=args.dropout, activation=getattr(nn, args.activation))
            optim = getattr(tt.optim, args.optimizer)
            model = CoxTime(net, optim, device=args.device, labtrans=labtrans)
            model.label_transform = labtrans

            y_train = model.label_transform.fit_transform(*(t_train, e_train))
            y_val = model.label_transform.transform(*(t_val, e_val))

            val = (x_val, y_val)
            val_size = x_val.shape[0]

            model.optimizer.set_lr(args.lr)
            model.optimizer.set('weight_decay', args.weight_decay)
            if args.early_stop:
                callbacks = [tt.callbacks.EarlyStopping()]
            else:
                callbacks = None
            model.fit(input=x_train, target=y_train, batch_size=args.batch_size, epochs=args.n_epochs,
                      callbacks=callbacks, verbose=args.verbose, val_data=val, val_batch_size=val_size)
            model.compute_baseline_hazards()
            surv_df = model.predict_surv_df(x_test)
            time_coordinates = surv_df.index.values
            surv_test = surv_df.values.T

            # add the initial time point
            time_coordinates = np.concatenate([np.array([0]), time_coordinates], 0)
            surv_test = np.concatenate([np.ones([surv_test.shape[0], 1]), surv_test], 1)
        elif args.model == "CoxNet":
            y_train_val = np.empty(dtype=[('cens', bool), ('time', np.float64)], shape=t_train_val.shape[0])
            y_train_val['cens'] = e_train_val
            y_train_val['time'] = t_train_val

            model = CoxnetSurvivalAnalysis(l1_ratio=0.5, n_alphas=100, fit_baseline_model=True)
            model.fit(x_train_val, y_train_val)
            pred_surv = model.predict_survival_function(x_test)
            surv_test, time_coordinates = format_pred_sksurv(pred_surv)
        elif args.model == "RSF":
            y_train_val = np.empty(dtype=[('cens', bool), ('time', np.float64)], shape=t_train_val.shape[0])
            y_train_val['cens'] = e_train_val
            y_train_val['time'] = t_train_val

            model = RandomSurvivalForest(n_estimators=100, n_jobs=None, random_state=seed_i)
            model.fit(x_train_val, y_train_val)
            pred_surv = model.predict_survival_function(x_test)
            surv_test, time_coordinates = format_pred_sksurv(pred_surv)
        elif args.model == "GB":
            y_train_val = np.empty(dtype=[('cens', bool), ('time', np.float64)], shape=t_train_val.shape[0])
            y_train_val['cens'] = e_train_val
            y_train_val['time'] = t_train_val

            model = ComponentwiseGradientBoostingSurvivalAnalysis(loss='coxph', n_estimators=100, random_state=seed_i)
            model.fit(x_train_val, y_train_val)
            pred_surv = model.predict_survival_function(x_test)
            surv_test, time_coordinates = format_pred_sksurv(pred_surv)
        elif args.model == "IWSG":
            n_bins = len(discrete_bins_e)
            Fmodel = DiscNN(
                n_features=n_features,
                output_size=n_bins,
                hidden_size=args.neurons,
                norm=args.norm,
                activation=args.activation,
                dropout=args.dropout,
            )
            Gmodel = DiscNN(
                n_features=n_features,
                output_size=n_bins,
                hidden_size=args.neurons,
                norm=args.norm,
                activation=args.activation,
                dropout=args.dropout,
            )
            # remove the first time point in the bins for training, because no one is before this point.
            Fmodel, Gmodel = train_iwsg(
                data_train, data_val, Fmodel, Gmodel, bins=discrete_bins_e[1:], optimizer=args.optimizer,
                batch_size=args.batch_size, epochs=args.n_epochs, lr=args.lr, lr_min=1e-3 * args.lr,
                weight_decay=args.weight_decay, device=device, early_stop=args.early_stop,
                fname=folder + f'/IWSG_{args.timestamp}', verbose=args.verbose)
            x_test = torch.from_numpy(x_test).float().to(device)
            surv_test = make_survival_prediction(x_test, n_bins, Fmodel, device)
            # the surv curves are starting from 0, so we need to add the first bin
            # (remember previously we remove this first time point so we can now replace it with 0)
            discrete_bins_e[0] = 0
            time_coordinates = discrete_bins_e
        else:
            raise ValueError(f"Unknown model name: {args.model}")

        # evaluate the performance
        if args.model != "CQRNN":
            if isinstance(time_coordinates, torch.Tensor):
                time_coordinates = time_coordinates.cpu().numpy()
            if isinstance(surv_test, torch.Tensor):
                surv_test = surv_test.cpu().numpy()
            evaler = SurvivalEvaluator(surv_test, time_coordinates, t_test, e_test, t_train_val, e_train_val,
            
                                       predict_time_method="Median", interpolation='Linear')
            """
            fig, ax = plt.subplots(figsize=(6, 4))
            event_idx = np.where(e_test == 1)[0][0:2]
            #for j, idx in enumerate(event_idx):
                #ax = plt.plot(time_coordinates, 1 - surv_test[idx], label=f'Failure #{j + 1}')

            censor_idx = np.where(e_test == 0)[0][0:3]
            for j, idx in enumerate(censor_idx):
                ax = plt.plot(time_coordinates, 1 - surv_test[idx], linestyle='--', label=f'Censor #{j + 1}')
            ax = plt.xlabel("Month")
            ax = plt.ylabel("Urethroplasty Failure Probability")
            plt.legend()
            fig.savefig(f'{path}/{args.model}_{i}.png', dpi=400)
            """
        else:
            if isinstance(quan_levels, torch.Tensor):
                quan_levels = quan_levels.cpu().numpy()
            if isinstance(quan_test, torch.Tensor):
                quan_test = quan_test.cpu().numpy()
            evaler = QuantileRegEvaluator(quan_test, quan_levels, t_test, e_test, t_train_val, e_train_val,
                                          predict_time_method="Median", interpolation='Linear')
            fig, ax = plt.subplots(figsize=(6, 4))
            event_idx = np.where(e_test == 1)[0][0:2]
            for j, idx in enumerate(event_idx):
                ax = plt.plot(quan_test[idx], quan_levels, label=f'Failure #{j + 1}')
            censor_idx = np.where(e_test == 0)[0][0:3]
            for j, idx in enumerate(censor_idx):
                ax = plt.plot(quan_test[idx], quan_levels, linestyle='--', label=f'Censor #{j + 1}')
            ax = plt.xlabel("Month")
            ax = plt.ylabel("Urethroplasty Failure Probability")
            plt.legend()
            fig.savefig(f'{path}/{args.model}_{i}.png', dpi=400)

        c_index = evaler.concordance(ties="Risk")[0]
        #wandb.log({'C-index': c_index, 'epoch': epoch})
        ibs_score = evaler.integrated_brier_score(num_points=10, IPCW_weighted=False)
        hinge_abs = evaler.mae(method='Hinge', verbose=False, weighted=True)
        po_abs = evaler.mae(method='Pseudo_obs', verbose=False, weighted=False)
        km_cal_score = evaler.km_calibration()
        p_value, _ = evaler.d_calibration()
        brier_score = evaler.brier_score(180, IPCW_weighted=False)
        auroc = evaler.auc(180)  #When does she want to evaluate the AUC? currently 24 days might change




        ci.append(c_index)
        ibs.append(ibs_score)
        mae_hinge.append(hinge_abs)
        mae_po.append(po_abs)
        km_cal.append(km_cal_score)
        d_cal.append(p_value)
        bs.append(brier_score)
        auc.append(auroc)

        
        wandb.log({'C-index': c_index,
                   'IBS': ibs_score,
                   'MAE_Hinge': hinge_abs,
                   'MAE_PO': po_abs,
                   'KM-cal': km_cal_score,
                   'D-cal': p_value,
                   'BS': brier_score,
                   'AUC': auroc})

        """
        
        print(f"Fold {i+1} Results:")
        print(f"  C-index: {c_index:.4f}")
        print(f"  IBS: {ibs_score:.4f}")
        print(f"  MAE_Hinge: {hinge_abs:.4f}")
        print(f"  MAE_PO: {po_abs:.4f}")
        print(f"  KM-cal: {km_cal_score:.4f}")
        print(f"  D-cal: {p_value:.4f}")
        print(f"  BS: {brier_score:.4f}")
        print(f"  AUC: {auroc:.4f}\n")
        """
            
        """
        print_performance(
            path=path,
            Cindex=ci,
            IBS=ibs,
            MAE_Hinge=mae_hinge,
            MAE_PO=mae_po,
            KM_cal=km_cal,
            D_cal=d_cal,
        )
        """


if __name__ == '__main__':
    # enable for debugging
    # torch.autograd.set_detect_anomaly(True)

    args = generate_parser()
    main(args)
    wandb.finish()
