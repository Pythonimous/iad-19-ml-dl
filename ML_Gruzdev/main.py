import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # only way to kill excessive torch warnings

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import parse_args_main
from util import load_data, get_pos_map
from model import TransformSentence, SentencesDataset, LSTMTagger
from train import train_model, test_model, demonstrate


def data_from_folder(data_directory):
    """Load all necessary data and lookups.

    :param
    data_directory (string): PATH to data;

    :returns
    data_loaders (dict): Dictionary containing train, validation and test loaders
    hanzi (list): Hanzi lookup list
    pos_map (dict): POS map
    raw_test (list): Test data in conllu format for demonstration.

    """

    raw_train, raw_val, raw_test, hanzi = load_data(data_directory)
    pos_map = get_pos_map(raw_train)

    preprocess_sentence = TransformSentence(hanzi, pos_map)
    data_train = SentencesDataset(raw_train, transform=preprocess_sentence)
    data_val = SentencesDataset(raw_val, transform=preprocess_sentence)
    data_test = SentencesDataset(raw_test, transform=preprocess_sentence)

    ''' We'll be using batch size 1 because all sentences are different length,
        and we can't effectively resize them, unlike images (to e.g. crop) '''

    train_loader = DataLoader(data_train, num_workers=0, batch_size=1, shuffle=True)
    val_loader = DataLoader(data_val, num_workers=0, batch_size=1, shuffle=True)
    test_loader = DataLoader(data_test, num_workers=0, batch_size=1, shuffle=True)
    data_loaders = {'train': train_loader,
                    'validation': val_loader,
                    'test': test_loader}
    return data_loaders, hanzi, pos_map, raw_test


def save_params(arguments):
    """ Write launch arguments to text file.

    :param
    arguments (Namespace): command line arguments

    """

    os.makedirs(arguments.save_dir, exist_ok=True)
    param_file = f'{arguments.save_dir}/params.txt'
    with open(param_file, 'w') as p_out:
        p_out.write('\n'.join(sys.argv[1:]))


def main():
    global args
    args = parse_args_main()
    save_params(args)
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    print(f'\nYour arguments: {args}')
    torch.manual_seed(args.seed)

    data_loaders, hanzi, pos_map, raw_test = data_from_folder(args.data_dir)

    EMBEDDING_DIM = 4001
    HIDDEN_DIM = 64               # Different configurations might work,
    TAGSET_SIZE = len(pos_map)    # but it is this toy config that yielded
    N_LAYERS = args.layers        # the best results on this representation
    BIDIRECTIONAL = args.bidirectional
    DROPOUT = args.dropout

    model = LSTMTagger(EMBEDDING_DIM,
                       HIDDEN_DIM,
                       TAGSET_SIZE,
                       N_LAYERS,
                       BIDIRECTIONAL,
                       DROPOUT)

    device = torch.device("cuda" if args.use_gpu else "cpu")
    model.to(device)

    if args.reload:
        if os.path.isfile(args.reload):
            print(f"Loading checkpoint '{args.reload}'")
            checkpoint = torch.load(args.reload)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.reload_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{args.reload}'"
                  f"(epoch {checkpoint['epoch']},"
                  f"accuracy {checkpoint['best_acc']})")
        else:
            print(f"No checkpoint found at '{args.reload}'")

    if args.test:
        test_model(model, data_loaders['test'], use_gpu=args.use_gpu)

    elif args.demonstrate:
        demonstrate(model, raw_test, hanzi, pos_map, use_gpu=args.use_gpu)

    else:
        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        # Decay LR by a factor of gamma every step_size epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.step_size,
                                                    gamma=args.gamma)
        print('Beginning to train')
        model = train_model(model, data_loaders,
                            criterion, optimizer, scheduler,
                            args.save_dir, num_epochs=5)


if __name__ == '__main__':
    main()
