#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
sns.set_theme(context='talk')

# use ggplot style for better and more beautiful plots
#plt.style.use('seaborn-talk')

colors = ['#b83945', '#377483', '#e3e457', '#4f845c']
light_colors = ['#fbdfe2', '#c7dff0', '#fcfce3', '#cfe7c4']


risk_factors = [
         "smoker_status"       
]
risk_factors_group = [
    ['Non-Smoker', 'Smoker'],
     
    ]

# load datasets
#version = "v1"
data = pd.read_excel(f"/Users/preciousajilore/Documents/GitHub/torchmtlr/notebooks/new_data_with_smoker.xlsx")
data["datetofailureorfollowup"] = pd.to_numeric(data["datetofailureorfollowup"], errors="coerce")
data["Failure "] = pd.to_numeric(data["Failure "], errors="coerce")
#external_valset = pd.read_csv(f"data/ATP_BCGP/{version}/BCGP_preprocessed_onehot_50.csv")

#This is for if we want to run multiple datasets
#datasets = {"ATP": internal_data,
#            "BCGP": external_valset}


#print(f"Dataset: {data_name}")
for i, risk in enumerate(risk_factors):
        # set the figure size
        plt.figure(figsize=(10, 10))
        #
        # # fill the missing values with the median
        # data[risk] = data[risk].fillna(data[risk].median())
        group_name = risk_factors_group[i]
        print("-" * 50)
        is_nunique = data[risk].nunique()
        if is_nunique == 2:
            lower_group = data[data[risk] == 0]
            print(f"Lower group: {len(lower_group)}")
            upper_group = data[data[risk] == 1]
            print(f"Upper group: {len(upper_group)}")
            lower_group.loc[:, "Failure "] = pd.to_numeric(lower_group["Failure "], errors="coerce")
            upper_group.loc[:, "Failure "] = pd.to_numeric(upper_group["Failure "], errors="coerce")

            lower_group = lower_group.dropna(subset=["datetofailureorfollowup", "Failure "])
            upper_group = upper_group.dropna(subset=["datetofailureorfollowup", "Failure "])

        elif is_nunique == 3:
            lower_group = data[data[risk] == 0]
            mid_group = data[data[risk] == 1]
            upper_group = data[data[risk] == 2]
            
            lower_group.loc[:, "Failure "] = pd.to_numeric(lower_group["Failure "], errors="coerce")
            mid_group.loc[:, "Failure "] = pd.to_numeric(mid_group["Failure "], errors="coerce")
            upper_group.loc[:, "Failure "] = pd.to_numeric(upper_group["Failure "], errors="coerce")

            lower_group = lower_group.dropna(subset=["datetofailureorfollowup", "Failure "])
            mid_group = mid_group.dropna(subset=["datetofailureorfollowup", "Failure "])
            upper_group = upper_group.dropna(subset=["datetofailureorfollowup", "Failure "])


        else:
            median = data[risk].median()
            print(f"Median of {risk}: {median}")

            lower_group = data[data[risk] <= median]
            print(f"Lower group: {len(lower_group)}")
            upper_group = data[data[risk] > median]
            print(f"Upper group: {len(upper_group)}")
        
        



        T = data['datetofailureorfollowup']
        E = data['Failure ']
        groups = data[risk]

        # Log-rank test across all groups
        #results = logrank_test(T, groups=groups, event_observed=E)
        results = logrank_test(lower_group["datetofailureorfollowup"], upper_group["datetofailureorfollowup"], lower_group["Failure "], upper_group["Failure "])
        results.print_summary()
        p_val = results.p_value

        # Kaplan-Meier curve
        low_group = KaplanMeierFitter().fit(lower_group["datetofailureorfollowup"], event_observed=lower_group["Failure "], label=group_name[0])
        #mid_group = KaplanMeierFitter().fit(mid_group["datetofailureorfollowup"], event_observed=mid_group["Failure "], label=group_name[1])
        upper_group = KaplanMeierFitter().fit(upper_group["datetofailureorfollowup"], event_observed=upper_group["Failure "], label=group_name[1])

        low_group.plot_survival_function(color=colors[0])
        #mid_group.plot_survival_function(color=colors[1])
        upper_group.plot_survival_function(color=colors[1])

        plt.xlabel('Months', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title(f'$P={p_val:3f}$', fontsize=14)
        plt.tight_layout()
        plt.legend(loc="upper left", prop={'size': 11})
        plt.savefig(f'{risk}.png', dpi=400)

        plt.close()
