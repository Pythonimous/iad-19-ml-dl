import argparse


def parse_args_main():
    """Arguments for main.py"""
    parser = argparse.ArgumentParser(description='PyTorch Chinese POS Tagger on hieroglyphic features. '
                                                 'I am so bad at my job it hurts.')
    parser.add_argument('--use_gpu', default=False, action='store_true',
                        help='use gpu for computation')

    parser.add_argument('--data_dir', default='data/', metavar='PATH',
                        help='set directory for conllu Chinese data')
    parser.add_argument('--save_dir', default='checkpoints/', metavar='PATH',
                        help='set directory to save models')

    parser.add_argument('--reload', default='', metavar='PATH',
                        help='path to checkpoint to load (default: none)')
    parser.add_argument('--test', default=False, action='store_true',
                        help='test model on test set (use with --reload)')
    parser.add_argument('--demonstrate', default=False, action='store_true',
                        help='test your luck! classify a random sentence from the test set '
                             '(use with --reload instead of --test)')

    parser.add_argument('--bidirectional', default=False, action='store_true',
                        help='you could use this, but is it worth it?')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of layers on lstm')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='set dropout rate')

    parser.add_argument('--epochs', type=int, default=1,
                        help='number of total epochs to run')

    parser.add_argument('--lr', type=float, default=0.1,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--step_size', type=int, default=10,
                        metavar='N', help='decay learning rate every N epochs')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='decay learning rate by a factor of gamma')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42, what else did you expect?)')

    args = parser.parse_args()
    return args


def parse_args_get_data():
    """Arguments for get_data.py"""
    parser = argparse.ArgumentParser(description='Chinese data downloader.')
    parser.add_argument('--data_dir', default='data/', metavar='PATH',
                        help='set directory for train, test and val for conllu-formated Chinese data')
    args = parser.parse_args()
    return args
