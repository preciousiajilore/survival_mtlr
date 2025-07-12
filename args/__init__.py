import argparse


def str_to_bool(arg):
    """Convert an argument string into its boolean value.

    Args:
        arg: String representing a bool.

    Returns:
        Boolean value for the string.
    """
    if arg.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str_to_list(arg):
    """Convert an argument string into a list.

    Args:
        arg: String representing a list.

    Returns:
        List value for the string.
    """
    if arg is None:
        return []
    else:
        return [int(x) for x in arg.split(',')]


def generate_parser():
    parser = argparse.ArgumentParser(description="Argument parser for the experiment.")

    # --------------------------------
    # General experiment parameters
    parser.add_argument('--n_exp', type=int, default=10,
                        help="Number of experiments to run.")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed.")
    parser.add_argument('--model', type=str, default="MTLR",
                        choices=["CoxNet", "RSF", "GB", "DeepHit", "CoxTime", "Nnet-survival", "IWSG",
                                 "CoxPH", "MTLR", "CQRNN", "WeibullAFT", "LogLogisticAFT",
                                 "CoxPH-LDR", "MTLR-LDR", "WeibullAFT-LDR", "LogLogisticAFT-LDR",
                                 ],
                        help="Model name.")
    parser.add_argument('--fs', type=str, default='lasso_cox',
                        choices=['None', 'cox_score_test', 'lasso_cox', ],
                        help="Feature selection method.")
    # --------------------------------
    # LDR parameters.
    parser.add_argument('--alpha', type=float, default=10, # >= 0.1 e.g., [0.1, 1, 10, 100]
                        help="Balance weight for the Orthogonality term.")
    parser.add_argument('--beta', type=float, default=0.01, # <= 0.1 e.g., [0.1, 0.01, 0.001]
                        help="Balance weight for the IPM term.")
    parser.add_argument('--ipm', type=str, default='mmd-rbf',
                        help="IPM function. Possible values: 'mmd-lin', 'mmd-rbf'")
    parser.add_argument('--d_dims', type=str_to_list, default=[],
                        help="Hidden neurons of the distribution networks. No space between numbers.")

    # --------------------------------
    # Network Structure parameters. Used for CoxPH, MTLR, DeepHit, CQRNN, LogNormalNN only.
    parser.add_argument('--neurons', type=str_to_list, default=[],
                        help="Hidden neurons in neural network. No space between numbers.")
    parser.add_argument('--norm', type=str_to_bool, default=True,
                        help="Whether to use batch norm in neural network.")
    parser.add_argument('--dropout', type=float, default=0.4,
                        help="Dropout rate.")
    parser.add_argument('--activation', type=str, default='ReLU',
                        help="Activation function. Possible values: Sigmoid, Tanh, ReLU, LeakyReLU, PReLU, ELU, SELU,"
                             "See https://pytorch.org/docs/stable/nn.html for more choices.")

    # --------------------------------
    # Training parameters, used for CoxPH, MTLR, DeepHit, CQRNN, LogNormalNN only.
    parser.add_argument('--optimizer', type=str, default="AdamW",
                        help="Optimizer for training. Possible values: 'SGD', 'Adam', ''AdamW', etc.")
    parser.add_argument('--n_epochs', type=int, default=10000,
                        help="Maximum number of training epochs. ")
    parser.add_argument('--early_stop', type=str_to_bool, default=True,
                        help="Whether to use early stop during training.")
    parser.add_argument('--batch_size', type=int, default=1024,
                        help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help="Term for weight decay (L2 penalty).")

    # --------------------------------
    # Display parameters, used for CoxPH, MTLR, DeepHit, CQRNN, LogNormalNN only.
    parser.add_argument('--verbose', type=str_to_bool, default=True,
                        help="Whether to print training information.")

    args = parser.parse_args()
    return args
