import pickle
import os
import requests
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
from config import parse_args_get_data


args = parse_args_get_data()
folder = args.data_dir

if not os.path.isdir(folder):
    os.mkdir(folder)
if not os.path.isfile(f'{folder}/zh_gsd-ud-train.conllu'):
    train_u = 'https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSD/master/zh_gsd-ud-train.conllu'
    urlretrieve(train_u, filename=f'{folder}/zh_gsd-ud-train.conllu')

if not os.path.isfile(f'{folder}/zh_gsd-ud-dev.conllu'):
    val_u = 'https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSD/master/zh_gsd-ud-dev.conllu'
    urlretrieve(val_u, filename=f'{folder}/zh_gsd-ud-dev.conllu')

if not os.path.isfile(f'{folder}/zh_gsd-ud-test.conllu'):
    test_u = 'https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSD/master/zh_gsd-ud-test.conllu'
    urlretrieve(test_u, filename=f'{folder}/zh_gsd-ud-test.conllu')

if not os.path.isfile('data/hanzi.pkl'):
    sess = requests.Session()
    hanzi_page = sess.get('http://www.hanzicraft.com/lists/frequency')
    soup = BeautifulSoup(hanzi_page.text, 'html.parser')
    hanzi = [ch.text.strip().split('\n')[0]
             for ch in soup.findAll('li', class_='list')][:4000]  # Top 4000 characters only
    with open('data/hanzi.pkl', 'wb') as h:
        pickle.dump(hanzi, h)
    h.close()
