import numpy as np
import pandas as pd
from tqdm import tqdm
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
import warnings
import matplotlib.pyplot as plt


def unicox_fs(
        train,
        val,
        test,
        verbose: bool = True,
):
    features = train.columns.to_list()
    features.remove('time')
    features.remove('event')

    selected_features = []
    for feature in features:
        if len(train[feature].unique()) == 1:
            if verbose:
                print(f"Feature {feature} has only one unique value, skipping.")
            continue
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train[[feature, 'time', 'event']], duration_col='time', event_col='event')
        if verbose:
            print(f"Feature: {feature}")
            print(f"Coefficient: {cph.params_[feature]}")
            print(f"P-value: {cph.summary.loc[feature, 'p']}")
        # cph.print_summary()
        if cph.summary.loc[feature, 'p'] < 0.05:
            selected_features.append(feature)
    if verbose:
        print(f"Selected features: {selected_features}")
    selected_features.append('time')
    selected_features.append('event')
    return train[selected_features], val[selected_features], test[selected_features]


def coxen_fs(
        train,
        val,
        test,
        verbose: bool = True,
        plot: bool = False
):
    l1_ratio_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]

    features = train.columns.to_list()
    features.remove('time')
    features.remove('event')

    X = train[features]
    event_times = train['time'].values
    event_indicators = train['event'].values.astype('bool')

    y = np.empty(dtype=[('cens', bool), ('time', np.float64)], shape=event_times.shape[0])
    y['cens'] = event_indicators
    y['time'] = event_times

    best_cindex = 0
    for l1_ratio in tqdm(l1_ratio_list, total=len(l1_ratio_list), disable=verbose):
        coxnet_pipe = make_pipeline(
            CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio='auto', max_iter=1000)
        )
        warnings.simplefilter("ignore", UserWarning)
        coxnet_pipe.fit(X, y)

        estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        gcv = GridSearchCV(
            make_pipeline(CoxnetSurvivalAnalysis(l1_ratio=l1_ratio)),
            param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
            cv=cv,
            error_score=0.5,
            n_jobs=4).fit(X, y)

        if gcv.best_score_ > best_cindex:
            best_cindex = gcv.best_score_
            best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
    # best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
    if verbose:
        print("ElasticNetCox: Chosen alpha and l1 ratio are: {} and {}".format(best_model.alphas[0], best_model.l1_ratio))
    best_coefs = pd.DataFrame(
        best_model.coef_,
        index=train.columns.drop(['time', 'event']),
        columns=["coefficient"]
    )

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    if verbose:
        print("Number of non-zero coefficients: {}".format(non_zero))

    non_zero_coefs = best_coefs.query("coefficient != 0")
    if plot:
        coef_order = non_zero_coefs.abs().sort_values("coefficient").index

        fig, ax = plt.subplots(figsize=(6, 6))
        non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
        ax.set_xlabel("coefficient")
        ax.grid(True)
        fig.subplots_adjust(left=0.5)
        fig.savefig("Coxen_coef.png", dpi=300)
    if verbose:
        print("Feature selected: {}".format(non_zero_coefs))

    fs_columns = non_zero_coefs.index.values.tolist()
    fs_columns.append('time')
    fs_columns.append('event')
    return train[fs_columns], val[fs_columns], test[fs_columns]
