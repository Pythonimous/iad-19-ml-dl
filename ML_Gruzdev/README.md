# Chinese Parts-of-Speech tagger on binary features

The purpose of this program was to learn project structure and practice in end-to-end project building. For academic purposes, we have created a simple parts of speech tagger based on a long short term memory model using simple binary hieroglyphic features.

## Organisation
The code in the repository is organized as follows:
  - main.py: driver code;
  - model.py: LSTM model and related functions;
  - train.py: functions related to training and testing;
  - util.py: utility functions

# Usage

## Install requirements
```bash
pip3 install -r requirements.txt

```

## Download data
Use get_data.py to download necessary data in conllu format + hanzi (hieroglyph) lookup list.

```bash
usage: get_data.py [-h] [--data_dir PATH]

Chinese data downloader.

optional arguments:
  -h, --help       show this help message and exit
  --data_dir PATH  set directory for train, test and val for conllu-formated
                   Chinese data

```
## Training / testing
```bash
usage: main.py [-h] [--use_gpu] [--data_dir PATH] [--save_dir PATH]
               [--reload PATH] [--test] [--demonstrate] [--bidirectional]
               [--layers LAYERS] [--dropout DROPOUT] [--epochs EPOCHS]
               [--lr LR] [--step_size N] [--gamma GAMMA] [--seed SEED]

PyTorch Chinese POS Tagger on hieroglyphic features.

optional arguments:
  -h, --help         show this help message and exit
  --use_gpu          use gpu for computation
  --data_dir PATH    set directory for conllu Chinese data
  --save_dir PATH    set directory to save models
  --reload PATH      path to checkpoint to load (default: none)
  --test             test model on test set (use with --reload)
  --demonstrate      classify a random sentence from the test set (use with --reload instead of --test)
  --bidirectional    you could use this, but is it worth it?
  --layers LAYERS    number of layers on lstm
  --dropout DROPOUT  set dropout rate
  --epochs EPOCHS    number of total epochs to run
  --lr LR            initial learning rate
  --step_size N      decay learning rate every N epochs
  --gamma GAMMA      decay learning rate by a factor of gamma
  --seed SEED        random seed (default: 42)

```
