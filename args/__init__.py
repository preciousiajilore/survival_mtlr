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
    parser.add_argument('--n_exp', type=int, default=1,
                        help="Number of experiments to run.")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed.")
    parser.add_argument('--model', type=str, default="GB",
                        choices=["MTLR", "DeepHit", "CoxPH", "AFT", "GB", "RSF", "CoxNet",
                                 "CoxTime", "CQRNN", "LogNormalNN", ],
                        help="Model name.")
    # --------------------------------
    # Network Structure parameters. Used for CoxPH, MTLR, DeepHit, CQRNN, LogNormalNN only.
    parser.add_argument('--neurons', type=str_to_list, default=[32,64],
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
    parser.add_argument('--n_epochs', type=int, default=5,
                        help="Maximum number of training epochs. ")
    parser.add_argument('--early_stop', type=str_to_bool, default=True,
                        help="Whether to use early stop during training.")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=1,
                        help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help="Term for weight decay (L2 penalty).")

    # --------------------------------
    # Display parameters, used for CoxPH, MTLR, DeepHit, CQRNN, LogNormalNN only.
    parser.add_argument('--verbose', type=str_to_bool, default=True,
                        help="Whether to print training information.")

    args = parser.parse_args()
    return args
