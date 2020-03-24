#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip3 install conllu')


# ## Фичи

# В первую очередь используем иероглифы, которые, как известно, имеют в себе семантический компонент, и, таким образом, являются значимыми признаками. Список иероглифов извлечём с http://www.hanzicraft.com/lists/frequency - там 8943 наиболее распространённых иероглифов. (в Python своего встроенного списка нет)

# In[ ]:


from bs4 import BeautifulSoup
import requests
sess = requests.Session()


# Возьмём четыре тысячи наиболее частых иероглифов (на сайте они выстроены по частоте), потому что все сразу брать смысла нет.

# In[ ]:


page = sess.get('http://www.hanzicraft.com/lists/frequency')
soup = BeautifulSoup(page.text, 'html.parser')
hanzi = [ch.text.strip().split('\n')[0] for ch in soup.findAll('li', class_='list')][:4000]


# In[ ]:


len(hanzi)


# In[ ]:


hanzi_set = set(hanzi)


# Загрузим данные.

# In[ ]:


from conllu import parse
import urllib


# In[ ]:


train_url = 'https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSD/master/zh_gsd-ud-train.conllu'
urllib.request.urlretrieve(train_url, filename = 'zh_gsd-ud-train.conllu')


# In[ ]:


with open("zh_gsd-ud-train.conllu", "r", encoding="utf-8") as f:
    cntr = f.read()
f.close()


# In[ ]:


chinese_train = parse(cntr)


# In[ ]:


chinese_train[0]


# Для обучения будем использовать bag of characters, поскольку для иероглифов это просто предельно важно.

# In[ ]:


chinese_train_words = [[w['form'] for w in i] for i in chinese_train]
chinese_train_postags = [[w['upostag'] for w in i] for i in chinese_train]
train_data = [(chinese_train_words[i], chinese_train_postags[i]) for i in range(len(chinese_train))]


# In[ ]:


chinese_train_words[3], chinese_train_postags[3], train_data[3]


# Для иероглифов используем bag of characters. В основном иероглифы в одном слове не будут встречаться дважды, и словарь иероглифов довольно ограничен, так что этот подход довольно эффективен.

# In[ ]:


def features_list(word):
    features = [0] * 4000
    for ch in word:
        if isinstance(ch, int) and len(features) == 4000: # в таком случае это не иероглиф, а число. достаточно единожды.
            features.append(1)     # для чисел отдельная категория и своя часть речи.
        elif (not isinstance(ch, int)) and len(features) == 4000:
            features.append(0)
            
        if ch in hanzi_set: # быстрее membership checking. если иероглиф есть в списке:
            ind = hanzi.index(ch) # соответсвтующая позиция - 1
            features[ind] = 1
    return features


# Итого получаем на выходе вектор размерностью 4001: 4000 иероглифов + число/не число

# In[ ]:


def prepare_sequence(seq): # слова как вектор с тем, какие иероглифы в нём содержатся
    idxs = [features_list(w) for w in seq]
    return torch.tensor(idxs, dtype=torch.float)


# In[ ]:


def prepare_tags(seq, tags_ix): # тэги по маппингу
    idxs = [tags_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Построим таблицу POS тэгов.

# In[ ]:


train_pos = [dic['upostag'] for dic in sum(chinese_train, [])] # какие части речи есть в тренировочном корпусе? их и метим


# In[ ]:


textlabels = list(set(train_pos))
tag_to_ix = {}
for i in range(len(textlabels)):
    tag_to_ix[textlabels[i]] = i


# In[ ]:


tag_to_ix


# Получим тестовые данные.

# In[ ]:


test_url = 'https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSD/master/zh_gsd-ud-test.conllu'
urllib.request.urlretrieve(test_url, filename = 'zh_gsd-ud-test.conllu')


# In[ ]:


with open("zh_gsd-ud-test.conllu", "r", encoding="utf-8") as f: #https://github.com/UniversalDependencies/UD_Chinese-GSD
    cntst = f.read()
f.close()


# In[ ]:


chinese_test = parse(cntst)


# In[ ]:


chinese_test_words = [[w['form'] for w in i] for i in chinese_test]
chinese_test_postags = [[w['upostag'] for w in i] for i in chinese_test]
test_data = [(chinese_test_words[i], chinese_test_postags[i]) for i in range(len(chinese_test))]


# In[ ]:


chinese_test_words[2], chinese_test_postags[2], test_data[2]


# Приступим к питорчу

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm


# In[ ]:


EMBEDDING_DIM = 4001
HIDDEN_DIM = 64
VOCAB_SIZE = 4001


# In[ ]:


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# In[ ]:


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# In[ ]:


with torch.no_grad(): # scores before training
    inputs = prepare_sequence(train_data[0][0])
    tag_scores = model(inputs)
    print(tag_scores)


# In[ ]:


epochs = 10


# In[ ]:


for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in tqdm(train_data):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence)
        targets = prepare_tags(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
    print('Epoch {} has passed'.format(str(epoch+1)))


# In[ ]:


with torch.no_grad():
    inputs = prepare_sequence(train_data[0][0])
    tag_scores = model(inputs)

    print(tag_scores[0]) # максимальное значение в тензоре соответствует предсказанной POS


# In[ ]:


tags_predicted = []
tags_true = []
for d in tqdm(test_data):
    with torch.no_grad():

        tags = prepare_tags(d[1], tag_to_ix)
        tags_true.append(tags.tolist())
        
        inputs = prepare_sequence(d[0])
        tag_scores = model(inputs).tolist()
        tags_encoded = [i.index(max(i)) for i in tag_scores] # максимальное значение в тензоре соответствует предсказанной POS
        tags_predicted.append(tags_encoded)


# In[ ]:


tags_predicted[0], tags_true[0]


# Для глобальной статистики мы можем и unnestн'уть списки. Это поможет определить, какие части речи определились правильно, а какие - нет.

# In[ ]:


tags_pred = sum(tags_predicted, [])


# In[ ]:


tags_actual = sum(tags_true, [])


# In[ ]:


from sklearn.metrics import f1_score, confusion_matrix, accuracy_score


# In[ ]:


f1_score(tags_actual, tags_pred, average='micro')


# In[ ]:


accuracy_score(tags_actual, tags_pred)


# 'PART': 0,
#  'SYM': 1,
#  'ADP': 2,
#  'ADJ': 3,
#  'NOUN': 4,
#  'CCONJ': 5,
#  'DET': 6,
#  'VERB': 7,
#  'PRON': 8,
#  'X': 9,
#  'NUM': 10,
#  'PUNCT': 11,
#  'ADV': 12,
#  'AUX': 13,
#  'PROPN': 14

# In[ ]:


confusion_matrix(tags_actual, tags_pred)


# Для пяти эпох результаты очень хорошие. Это говорит об эффективности модели Bag of Characters для сильно основанного на иероглифах китайского языка!

# In[ ]:




