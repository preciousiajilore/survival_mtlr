#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter

# use ggplot style for better and more beautiful plots
plt.style.use('seaborn-talk')

colors = ['#b83945', '#377483', '#e3e457', '#4f845c']
light_colors = ['#fbdfe2', '#c7dff0', '#fcfce3', '#cfe7c4']


risk_factors = [
    'CORE_AGE_AT_SURVEY_RECEIPT',
    'CORE_PM01_CIRRHOSIS',
    'CORE_AM02_STAND',
    'CORE_PM01_MS',
    'CORE_PM01_COPD',
    'CORE_AM01_HANDEDNESS_1.0',
    'CORE_AM01_HANDEDNESS_2.0',
    'CORE_WH13_MENOPAUSE_OCCURRENCE',
    'CORE_SU03_HAIR_1.0',
    'CORE_SU03_HAIR_2.0',
    'CORE_SU03_HAIR_3.0',
    'CORE_SU03_HAIR_4.0',
    'CORE_SU03_HAIR_5.0',
    'CORE_SU04_EYES_1.0',
    'CORE_SU04_EYES_2.0',
    'CORE_SU04_EYES_3.0',
    'CORE_SU04_EYES_4.0',
    'CORE_SU04_EYES_5.0',
    'CORE_SU04_EYES_6.0',
    'CORE_ET06_WORK_ETS_FREQ_V2',
    'CORE_WS01_WORKING_STATUS_PT',
    'CORE_WS01_WORKING_STATUS_FT',
    'CORE_WS01_WORKING_STATUS_RETIRED',
    'CORE_WH06_FIRST_PREG_AGE',
    'CORE_FDRCAN',
    'CORE_AU01_ALCOHOL_EVER',
    'CORE_TU01_EVER_SMOKED',
    # 'SMK_STATUS',
    'CORE_FA01_MARITAL_STATUS',
    'CORE_FM03_CANCER_M',
    'CORE_CHROCON',
    'CORE_WH15_HRT_EVER',
]
risk_factors_group = [
    ['Young', 'Elderly'],
    ['No cirrhosis', 'Cirrhosis'],
    ['No standing', 'Standing'],
    ['No MS', 'MS'],
    ['No COPD', 'COPD'],
    ['None Left-handed', 'Left-handed'],
    ['None Right-handed', 'Right-handed'],
    ['No menopause', 'Menopause'],
    ['None Blonde', 'Blonde'],
    ['None Red', 'Red'],
    ['None Light Brown', 'Light Brown'],
    ['None Dark Brown', 'Dark Brown'],
    ['None Black', 'Black'],
    ['None Amber', 'Amber'],
    ['None Blue', 'Blue'],
    ['None Brown', 'Brown'],
    ['None Grey', 'Grey'],
    ['None Green', 'Green'],
    ['None Hazel', 'Hazel'],
    ['Low Frequency', 'High Frequency'],
    ['Not working PT', 'Working PT'],
    ['Not working FT', 'Working FT'],
    ['Not retired', 'Retired'],
    ['Early pregnancy', 'Late pregnancy'],
    ['No BCa', 'BCa'],
    ['No alcohol', 'Alcohol'],
    ['Never smoked', 'Smoked'],
    # ['Frequent smoker', 'Non-frequent smoker', ],
    ['Not married', 'Married'],
    ['No BCa', 'BCa'],
    ['No Chronic Disease', 'Chronic Disease'],
    ['No HRT', 'HRT'],
    ]

# load datasets
version = "v1"
internal_data = pd.read_csv(f"data/ATP_BCGP/{version}/ATP_preprocessed_onehot_50.csv")
external_valset = pd.read_csv(f"data/ATP_BCGP/{version}/BCGP_preprocessed_onehot_50.csv")

datasets = {"ATP": internal_data,
            "BCGP": external_valset}

for data_name, data in datasets.items():
    print(f"Dataset: {data_name}")
    for i, risk in enumerate(risk_factors):
        # set the figure size
        plt.figure(figsize=(3.5, 3))
        #
        # # fill the missing values with the median
        # data[risk] = data[risk].fillna(data[risk].median())
        group_name = risk_factors_group[i]
        print("-" * 50)
        is_binary = data[risk].nunique() == 2
        if is_binary:
            lower_group = data[data[risk] == 0]
            print(f"Lower group: {len(lower_group)}")
            upper_group = data[data[risk] == 1]
            print(f"Upper group: {len(upper_group)}")
        else:
            median = data[risk].median()
            print(f"Median of {risk}: {median}")

            lower_group = data[data[risk] <= median]
            print(f"Lower group: {len(lower_group)}")
            upper_group = data[data[risk] > median]
            print(f"Upper group: {len(upper_group)}")

        # log-rank test
        results = logrank_test(lower_group.time, upper_group.time, lower_group.event, upper_group.event)
        results.print_summary()
        p_val = results.p_value

        # Kaplan-Meier curve
        low_group = KaplanMeierFitter().fit(lower_group.time, event_observed=lower_group.event, label=group_name[0])
        up_group = KaplanMeierFitter().fit(upper_group.time.values, event_observed=upper_group.event.values, label=group_name[1])

        low_group.plot_cumulative_density(color=colors[0])
        up_group.plot_cumulative_density(color=colors[1])

        plt.xlabel('Months', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title(f'$P={p_val:3f}$', fontsize=14)
        plt.tight_layout()
        plt.legend(loc="upper left", prop={'size': 11})
        plt.savefig(f'figs/{data_name}/{risk}.png', dpi=400)

        plt.close()
