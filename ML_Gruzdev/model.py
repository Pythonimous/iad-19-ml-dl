import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset


class LSTMTagger(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 tagset_size,
                 n_layers,
                 bidirectional,
                 dropout):

        super(LSTMTagger, self).__init__()

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0)
        self.hidden2tag = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, tagset_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence):
        embedded = self.dropout(sentence)
        lstm_out, _ = self.lstm(embedded)
        tag_space = self.hidden2tag(self.dropout(lstm_out))
        tag_scores = func.log_softmax(tag_space, dim=1)
        return tag_scores


class SentencesDataset(Dataset):
    """
        Chinese words dataset.
    Args:
        data: as downloaded by get_data();
        transform: a suitable transformation / pipeline of transformations
    """
    def __init__(self, data, transform=None):
        self.sentences = [[word['form'] for word in sentence] for sentence in data]
        self.labels = [[word['upostag'] for word in sentence] for sentence in data]
        self.transform = transform

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.labels[idx]
        sample = {'sentence': sentence, 'tags': tags}

        if self.transform:
            sample = self.transform(sample)

        return sample


class TransformSentence(object):
    """
        Present sentence as a vector of hanzi occurrences + integer.
        As hieroglyphs each carry semantical meaning, we try and
        represent words as bag of characters.
    Args:
        hanzi (list): list of hanzi to lookup;
        pos_map (dict): POS lookup from POS to indices
    """

    def __init__(self, hanzi, pos_map):
        self.hanzi = hanzi
        self.lookup_hanzi = set(hanzi)
        self.pos_map = pos_map

    def vectorize(self, word):
        """Converts Chinese token into a list of binary features

        :param
        word (list): a list of Chinese tokens

        :return:
        features (list): a vector of 4001 binary hieroglyphic features

        """

        features = [0] * 4000
        for character in word:
            if isinstance(character, int) and len(features) == 4000:
                features.append(1)  # Integers are a separate position in a vector
            elif (not isinstance(character, int)) and len(features) == 4000:
                features.append(0)

            if character in self.lookup_hanzi:
                position = self.hanzi.index(character)
                features[position] = 1
        return features

    def __call__(self, sample):
        """Vectorizes POS-tags and sentence

        :param sample (dict): a pair of {'sentence':x, 'tags':y}

        :return: dict(). dict['sentence'] - binary matrix; dict['tags'] - numerical vector

        """

        sentence, tags = sample['sentence'], sample['tags']
        sentence = [self.vectorize(word) for word in sentence]
        sentence = torch.tensor(sentence, dtype=torch.float)

        tags = [self.pos_map[pos] for pos in tags]
        tags = torch.tensor(tags, dtype=torch.long)

        return {'sentence': sentence, 'tags': tags}
