import wandb
import pprint
import argparse

from run import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for the hyper-parameter sweep.")

    parser.add_argument('--model', type=str, default="MTLR",
                        choices=["CoxNet", "RSF", "GB", # for these three, you don't need any hyper-parameters tuning, just get 1 run and report the results
                                 "DeepHit", "CoxTime", "Nnet-survival", "IWSG",
                                 "CoxPH", "MTLR", "CQRNN", "WeibullAFT", "LogLogisticAFT",
                                 "CoxPH-LDR", "MTLR-LDR", "WeibullAFT-LDR", "LogLogisticAFT-LDR",
                                 ],
                        help="Model name.")

    parser.add_argument('--sweep_num', type=int, default=24,
                        help="Sweep number for wandb.")
    parser.add_argument('--sweep_id', type=str, default=None,
                        help="Sweep ID for wandb. If not None, sweep_num will be ignored. "
                             "Otherwise, a new sweep will be created.")

    args = parser.parse_args()

    name = args.model

    parameters = {
                # optimizable parameters example:
                # 'parameter_name': {'values': list[value1, value2, ...]}'
                'weight_decay': {'values': [0.1, 0.01]},
                'neurons': {'values': [[64], [64, 64], [64, 64, 16],
                                       [32], [32, 32], [32, 32, 16]]},
                'dropout': {'values': [0.4, 0.6]},
                # non-optimizable parameters example:
                # 'parameter_name': {'value': value}
                'optimizer': {'value': 'AdamW'},
                'lr': {'value': 0.001},
                'batch_size': {'value': 256},
                'norm': {'value': True},
                'activation': {'value': 'ReLU'},
                'early_stop': {'value': True},
                'n_epochs': {'value': 10000},
                'verbose': {'value': True},
                'seed': {'value': 0},
                'n_exp': {'value': 10},
                'model': {'value': args.model},
            }
    if 'LDR' in args.model:
        parameters.update({
            'alpha': {'values': [0.1, 1, 10]},
            'beta': {'values': [0.001, 0.01, 0.1]},
            'ipm': {'values': ['mmd-lin', 'mmd-rbf']},
            'd_dims': {'value': [[], [16], [16, 16],]},
        })

    if args.sweep_id is None:
        sweep_config = {
            'method': 'grid', # grid, random, bayes
            'name': name,
            'metric': {
                'name': 'mean-Cindex',
                'goal': 'maximize'
            },
            'parameters': parameters,
        }

        pprint.pprint(sweep_config)

        sweep_id = wandb.sweep(sweep=sweep_config, project="UroLOGIC")
        wandb.agent(sweep_id, main, count=args.sweep_num)
    else:
        wandb.agent(args.sweep_id, function=main, project="UroLOGIC", count=args.sweep_num)
